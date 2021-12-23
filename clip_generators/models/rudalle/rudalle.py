import json
import random
import time
from pathlib import Path

import requests
from PIL import Image
from deep_translator import DeepL
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.image_prompts import ImagePrompts
from rudalle.pipelines import generate_images, super_resolution, cherry_pick_by_clip
from rudalle.utils import seed_everything

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

class RudalleGenerator:
    def __init__(self, emoji=False):
        self.device = 'cuda'
        self.emoji = emoji
        if emoji:
            self.dalle = get_rudalle_model('Emojich', pretrained=True, fp16=True, device=self.device)
        else:
            self.dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=self.device)

        self.realesrgan = get_realesrgan('x4', device=self.device)
        self.tokenizer = get_tokenizer()
        self.vae = get_vae(dwt=True).to(self.device)
        ruclip, self.ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
        self.ruclip = ruclip.to(self.device)

        self.translator = DeepL(api_key='3e7d1072-c50d-6644-8bbf-fe9534efc63a:fx', use_free_api=True, source='en',
                           target='ru')

        self._stop = False

    def stop(self):
        self._stop = True

    def generate(self, text, seed, out_dir: Path,  image_prompt_url: str = None, image_cut_top = 4, nb_images=3,):
        if text.startswith('--raw'):
            translated_text = text[len('--raw'):].strip()
        else:
            translated_text = self.translator.translate(text)  # output -> Weiter so, du bist großartig

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'hyperparameters.json').write_bytes(json.dumps({
            'seed': seed,
            'text': text,
            'translated_text': translated_text,
            'image_prompt_url': image_prompt_url,
            'image_cut_top': image_cut_top,
        }, ensure_ascii=False).encode('utf-8'))
        # text = 'проклятая еда'
        (out_dir / 'top').mkdir(parents=True, exist_ok=True)

        schedule = [
            (2048, 0.995, nb_images),
            (1536, 0.99, nb_images),
            (1024, 0.99, nb_images),
            (1024, 0.98, nb_images),
            (512, 0.97, nb_images),
            (384, 0.96, nb_images),
            (256, 0.95, nb_images),
            (128, 0.95, nb_images),
        ]
        image_prompt = None
        if image_prompt_url:
            prompt = Image.open(requests.get(image_prompt_url.strip(), stream=True).raw).resize((256, 256))
            prompt.save(out_dir / 'prompt.png')
            prompt.crop((0, 0, prompt.size[0], image_cut_top * 8)).save(out_dir / 'cropped_prompt.png')
            yield out_dir / 'cropped_prompt.png'
            image_prompt = ImagePrompts(prompt, {'up': image_cut_top, 'left': 0, 'right': 0, 'down': 0}, self.vae, self.device, crop_first=True)


        seed_everything(seed)
        pil_images = []
        scores = []
        j = 0
        self._stop = False
        #yield out_dir / f'all.png'
        for top_k, top_p, images_num in schedule:
            if self._stop:
                break
            _pil_images, _scores = generate_images(translated_text, self.tokenizer, self.dalle, self.vae, top_k=top_k,
                                                   images_num=images_num, top_p=top_p, image_prompts=image_prompt, use_cache=True)
            pil_images += _pil_images
            scores += _scores
            (out_dir / str(j)).mkdir(exist_ok=True, parents=True)

            for i, image in enumerate(_pil_images):
                image.save(out_dir / str(j) / f'{text}_{seed}_{i}.png')
            grid = image_grid(_pil_images, 1, images_num)
            grid.save(out_dir / f'{j}.png')
            yield out_dir / f'{j}.png'

            grid = image_grid(pil_images, j + 1, nb_images)
            grid.save(out_dir / f'all.png')
            j = j + 1
        yield None
        grid = image_grid(pil_images, len(pil_images) // nb_images, nb_images)
        grid.save(out_dir / f'all.png')

        top_images, clip_scores = cherry_pick_by_clip(pil_images, translated_text, self.ruclip, self.ruclip_processor, device=self.device, count=3)
        sr_images = super_resolution(top_images, self.realesrgan)
        for i, image in enumerate(sr_images):
            image.save(out_dir / 'top' / f'{text}_{seed}_{random.randint(0, 999)}_hd.png')
        grid = image_grid(sr_images, 1, 3)
        grid.save(out_dir / f'tops.png')
        grid.save(out_dir / f'../{time.time()}_{text}.png')
        yield out_dir / f'tops.png'


class RudalleDreamer:
    def __init__(self):
        ...
