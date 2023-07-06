import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_polling
from aiogram.types import Message, ContentType
from PIL import Image

from VGG_model import VGGStyleTransfer

logging.basicConfig(level=logging.INFO)

bot = Bot(token=os.getenv('TG_BOT_TOKEN'))
dp = Dispatcher(bot)

style_transfer_model = VGGStyleTransfer()

user_state = {}

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: Message):
    await message.reply("Привет! Отправьте мне два изображения, и я перенесу стиль с одного на другое.\n"
                        "Используйте команду /transfer_style перед отправкой двух изображений.")

@dp.message_handler(commands=['transfer_style'])
async def transfer_style_command_handler(message: Message):
    user_state[message.from_user.id] = 'content'
    await message.reply("Пожалуйста, отправьте сначала изображение, которое вы хотите изменить, а затем изображение стиля.")

@dp.message_handler(content_types=ContentType.PHOTO)
async def process_photo(message: types.Message):
    if message.from_user.id in user_state:
        user_folder = os.path.join('imgs', str(message.from_user.id))
        os.makedirs(user_folder, exist_ok=True)
        
        img_path = os.path.join(user_folder, f"{user_state[message.from_user.id]}.jpg")
        
        await bot.download_file_by_id(file_id=message.photo[-1].file_id, destination=img_path)

        if user_state[message.from_user.id] == 'content':
            user_state[message.from_user.id] = 'style'
            await message.reply("Изображение контента успешно загружено. Теперь отправьте изображение стиля.")
        elif user_state[message.from_user.id] == 'style':
            await message.reply("Изображение стиля успешно загружено. Теперь обрабатываю изображения...")
            user_state.pop(message.from_user.id)

            content_img_path = os.path.join(user_folder, "content.jpg")
            style_img_path = os.path.join(user_folder, "style.jpg")

            content_image = style_transfer_model.load_and_preprocess_image(content_img_path)
            style_image = style_transfer_model.load_and_preprocess_image(style_img_path)
            output_img_path = os.path.join(user_folder, "output.jpg")

            try:
                output_image = style_transfer_model.perform_style_transfer(content_image, style_image, output_img_path)
                with open(output_img_path, 'rb') as photo:
                    await message.reply_photo(photo)
                    await message.reply("Обработка изображений успешно завершена!")
            except Exception as e:
                await message.reply(f"Произошла ошибка при обработке изображения: {str(e)}")

if __name__ == "__main__":
   start_polling(dp, skip_updates=True)
