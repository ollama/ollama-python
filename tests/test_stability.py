import ollama
import unittest
import time
from tqdm import tqdm


model = "mistral"
gen_prompt = 'Write a short stories about cute llamas with up to 200 words.'
chat_prompt = [
            {"role": "system",
             "content": "You are a writer. You write paragraphs with a word count between 100 and 200 words. "
                        "Each paragraph starts on a new line. You must follow the details and concepts of the "
                        "story already written."},
            {"role": "user",
             "content": "Once upon a time, in the vast, lush valleys of the Andes Mountains, there lived a cute "
                        "llama named Lulu. Lulu was not just any llama; she was adorned with the softest, whitest "
                        "fur that shimmered under the sun’s glow. Her large, doe-like eyes sparkled with curiosity,"
                        "making her the jewel of her herd. Lulu spent her days grazing on the rich, green grass "
                        "that carpeted the mountainside, her playful nature making her a favorite among the other "
                        "llamas. "
                        ""
                        "Lulu's curiosity often led her on adventures far beyond the grazing fields. One sunny "
                        "morning, she spotted a colorful butterfly fluttering by and couldn’t resist following it."
                        " The butterfly led her through winding paths and over sparkling streams, deeper into the "
                        "heart of the mountains than Lulu had ever been. Along the way, she discovered new plants "
                        "and flowers, their vibrant colors and exotic scents a delightful surprise."
                        ""
                        "As the sun began to dip below the horizon, Lulu realized she had strayed far from home. "
                        "The once familiar landscape now seemed strange and daunting in the twilight. She started "
                        "to feel a twinge of fear, but her adventurous spirit urged her to press on. Determined to "
                        "find her way back, Lulu used the stars as her guide, remembering the constellations her "
                        "mother had taught her."
                        ""
                        "Her journey through the night was not lonely, for the mountains were alive with nocturnal"
                        " creatures. A wise old owl hooted from the branches of an ancient tree, offering Lulu"
                        " words of encouragement. “Trust your instincts,” he said, his eyes twinkling in the "
                        "moonlight. Lulu felt a surge of confidence, grateful for the unexpected guardian. "
                        ""
                        "As dawn broke, Lulu found herself in a part of the valley she had never seen before. "
                        "It was a hidden paradise, with a sparkling lake that mirrored the sky. The grass was "
                        "softer and the air filled with the sweet melody of birds. Lulu drank from the lake, "
                        "the water refreshing her tired body. It was then she realized that sometimes getting "
                        "lost leads to the most beautiful discoveries."
                        ""
                        "Her adventure had attracted the attention of other creatures in the valley. A friendly "
                        "alpaca named Ari approached her, intrigued by the stranger. Ari was enchanted by Lulu’s "
                        "gentle nature and her story of the night’s adventure. He offered to show her the way home,"
                        "and Lulu happily accepted, excited to have made a new friend."
                        ""
                        "Together, they traversed the valley, with Ari sharing stories of the hidden wonders that"
                        " lay tucked away in corners Lulu had never explored. She listened intently, her mind "
                        "alive with the possibilities of future adventures. As they neared her home, Lulu realized "
                        "how much she had missed her family, and her heart swelled at the thought of seeing "
                        "them again."
                        ""
                        "Lulu’s return was met with joyous celebrations. Her family had been worried but were now"
                        "overjoyed to see her safe. Lulu shared her adventure, her eyes gleaming with the thrill "
                        "of her journey. Her herd listened in awe, Lulu's bravery and curiosity inspiring them."
                        ""
                        "The story of Lulu’s adventure spread throughout the valley, and she became somewhat of a "
                        "legend. Young llamas would gather around her, eager to hear her tales and learn from her "
                        "experiences. Lulu, with her new-found wisdom, taught them the importance of curiosity, "
                        "courage, and the beauty of discovering the unknown."
                        ""
                        "Lulu continued to explore, but never again did she stray too far from home without "
                        "leaving a trail. Her adventures brought her many new friends and experiences, enriching"
                        " her life in ways she never imagined. And in the heart of the Andes, under the vast "
                        "expanse of the sky, Lulu the llama lived happily, her spirit forever wandering, "
                        "forever free."},
            {"role": "assistant",
             "content": "Thank you for the story so far."},
            {"role": "user",
             "content": "Write 10 more paragraphs."},
        ]


class TestAddFunction(unittest.TestCase):
    def test_c_100c_gen(self):
        generated = []
        for _ in tqdm(range(100)):
            response = ollama.generate(model=model, prompt=gen_prompt)
            output = response['response']
            generated.append(output)
        self.assertEqual(len(generated), 100)

    def test_d_100c_chat(self):
        generated = []
        for _ in tqdm(range(100)):
            response = ollama.chat(
                model=model,
                messages=chat_prompt
            )
            output = response['message']['content']
            generated.append(output)
        self.assertEqual(len(generated), 100)

    def test_a_45m_gen(self):
        generated = []
        start_time = time.time()
        duration = 45 * 60
        end_time = start_time + duration
        with tqdm(total=duration) as pbar:
            while time.time() < end_time:
                response = ollama.generate(model=model, prompt=gen_prompt)
                output = response['response']
                generated.append(output)
                current_time = time.time()
                elapsed_time = current_time - start_time
                pbar.update(elapsed_time - pbar.n)
        self.assertGreater((time.time() - start_time), duration)

    def test_b_45m_chat(self):
        generated = []
        start_time = time.time()
        duration = 45 * 60
        end_time = start_time + duration
        with tqdm(total=duration) as pbar:
            while time.time() < end_time:
                response = ollama.chat(
                    model=model,
                    messages=chat_prompt
                )
                output = response['message']['content']
                generated.append(output)
                current_time = time.time()
                elapsed_time = current_time - start_time
                pbar.update(elapsed_time - pbar.n)
        self.assertGreater((time.time() - start_time), duration)


unittest.main()

