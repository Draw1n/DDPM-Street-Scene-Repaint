from PIL import Image

def apply_mask(origin_image_path, mask_path, output_path):
    # 打开原始图像和mask图像
    origin_image = Image.open(origin_image_path)
    mask = Image.open(mask_path).convert('L')

    # 确保mask和原始图像大小一致
    mask = mask.resize(origin_image.size, Image.ANTIALIAS)

    # 将mask应用到原始图像
    result_image = Image.new('RGB', origin_image.size)
    result_image.paste(origin_image, mask=mask)

    # 保存结果
    result_image.save(output_path)
    print('masked!')
    return result_image

def fill_masked(origin_image_path, denoise_image_path, mask_path, output_path):
    # 打开结果图像、评估图像和mask
    origin_image = Image.open(origin_image_path)
    denoise_image = Image.open(denoise_image_path)
    mask = Image.open(mask_path).convert('L')

    # 获取mask的黑色区域坐标
    black_pixels = [(x, y) for x in range(mask.width) for y in range(mask.height) if mask.getpixel((x, y)) == 0]

    # 在结果图像中对应的位置填充评估图像的像素值
    for x, y in black_pixels:
        denoise_pixel = denoise_image.getpixel((x % denoise_image.width, y % denoise_image.height))
        origin_image.putpixel((x, y), denoise_pixel)

    # 保存结果
    origin_image.save(output_path)
    print('filled!')
    return origin_image