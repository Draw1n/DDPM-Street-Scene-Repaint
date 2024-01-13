from PIL import Image, ImageDraw, ImageFont

def iamge_frame(path = '/root/diffusers_main/'):
    # 创建一个空白的图相框
    frame_width = 420
    frame_height = 160
    frame_color = (255, 255, 255)
    frame = Image.new("RGB", (frame_width, frame_height), frame_color)
    draw = ImageDraw.Draw(frame)

    # 读取三张图片
    original_img = Image.open(path+'original_img.png')
    masked_img = Image.open(path+'masked_img.png')
    repaint_img = Image.open(path+'repaint_img.png')

    # 调整图片大小为128x128
    original_img = original_img.resize((128, 128))
    masked_img = masked_img.resize((128, 128))
    repaint_img = repaint_img.resize((128, 128))

    # 将三张图片组合在一起
    frame.paste(original_img, (10, 10))
    frame.paste(masked_img, (150, 10))
    frame.paste(repaint_img, (290, 10))

    # 在每张图片下方添加名称
    font_size = 24
    font_color = (0, 0, 0)

    # 使用默认字体并指定字体大小
    font = ImageFont.load_default()

    draw.text((35, 140), 'original_img', font=font, fill=font_color)
    draw.text((180, 140), 'masked_img', font=font, fill=font_color)
    draw.text((320, 140), 'repaint_img', font=font, fill=font_color)

    # 保存结果图像
    frame.save(path+'frame_img.png')

    # 显示结果图像（可选）
    frame.show()