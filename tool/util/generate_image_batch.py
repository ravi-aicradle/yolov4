import os

supported_image_formats = ('.png', '.jpg')


def get_image_batches(input_dir, batch_size=1):
    file_list = []
    for file in os.listdir(input_dir):
        if file.endswith(supported_image_formats):
            file_list.append(os.path.join(input_dir, file))
    final = [file_list[i * batch_size:(i + 1) * batch_size] for i in
             range((len(file_list) + batch_size - 1) // batch_size)]
    return final
