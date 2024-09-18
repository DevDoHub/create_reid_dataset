from ultralytics import YOLO
import cv2
import datetime
import os


"""
检测视频中出现人到人消失，分别保存成一个小视频
遍历一遍, 保存

1.视频中人连续2帧以上识别到才会保存一个小视频, 并且将人物裁剪出来保存

目前支持mp4格式的视频, 需要其他请自行修改

"""

# 加载YOLOv8模型
# model = YOLO('yolov8n.pt').to('cuda')  # 将模型加载到GPU上
model = YOLO('yolov8n.pt')  # 请使用适合的模型路径

# 输入视频文件路径
input_video_path = './test.mp4'  # 替换为你的输入视频路径
input_video_path = './v_id'  # 替换为你的输入目录路径

now = datetime.datetime.now()
output_folder_prefix = 'output_videos' + now.strftime("%Y%m%d%H%M%S")

# 判断input_video_path是视频文件还是目录
input_video_files = []
if not os.path.isdir(input_video_path):
    if input_video_path.endswith('.mp4'):
        input_video_files = [input_video_path]
else:
    # 获取目录下所有的视频文件路径
    # 目录下有目录的情况下，需要递归遍历

    for root, _, files in os.walk(input_video_path):
        for file in files:
            if file.endswith('.mp4'):
                input_video_files.append(os.path.join(root, file))

    # input_video_files = [os.path.join(input_video_path, f) for f in os.listdir(input_video_path) if f.endswith('.mp4')]

if not input_video_files:
    print(f"未找到任何视频文件！输入: {input_video_path}")
    exit()

os.makedirs(output_folder_prefix, exist_ok=True)


for input_video_file in input_video_files:
    
    output_v_folder = os.path.join(output_folder_prefix, "video_" + os.path.splitext(input_video_file)[0])
    output_i_folder = os.path.join(output_folder_prefix, "image_" + os.path.splitext(input_video_file)[0])

    # 创建输出文件夹
    os.makedirs(output_v_folder, exist_ok=True)
    os.makedirs(output_i_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化变量
    frame_index = 0
    recording = False  # 标记当前是否在记录视频
    out = None  # 视频写入器
    clip_index = 1  # 小视频编号

    # 遍历视频的每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLOv8进行检测
        results = model(frame)
        person_detected = False

        # 检查是否检测到人物
        for result in results:
            for box in result.boxes:
                if int(box.cls.item()) == 0:  # 0表示人物

                    # 获取边界框坐标并裁剪图像
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_person = frame[y1:y2, x1:x2]

                    _output_i_folder = os.path.join(output_i_folder, str(clip_index))
                    if not os.path.exists(_output_i_folder):
                        os.makedirs(_output_i_folder)

                    # 保存裁剪结果
                    crop_path = os.path.join(_output_i_folder, f'person_标签_{clip_index}_第{frame_index}帧_{x1}_{y1}.jpg')
                    # 128*256
                    cropped_person = cv2.resize(cropped_person, (128, 256))
                    cv2.imwrite(crop_path, cropped_person)

                    person_detected = True
                    break

        # 判断是否需要开始或停止录制
        if person_detected:
            if not recording:  # 如果当前没有录制并检测到人物，开始录制
                _output_v_folder = os.path.join(output_v_folder, f'person_clip_{clip_index}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(_output_v_folder, fourcc, fps, (frame_width, frame_height))
                recording = True
                print(f"开始录制小视频: {_output_v_folder}")

            # 写入当前帧
            out.write(frame)
        else:
            if recording:  # 如果当前正在录制但没有检测到人物，停止录制
                out.release()
                recording = False
                clip_index += 1
                print("停止录制并保存小视频")

        frame_index += 1

    # 释放资源
    if recording:
        out.release()  # 如果最后仍在录制，释放写入器

    cap.release()
    cv2.destroyAllWindows()
    print("视频处理完毕，所有小视频已保存！")
