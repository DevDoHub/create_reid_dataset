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
# input_video_path = './test.mp4'  # 替换为你的输入视频路径
input_video_path = './output_videos_20240920144211/video'  # 替换为你的输入目录路径

now = datetime.datetime.now()
output_i_folder = './output_videos_20240920144211/image'
os.makedirs(output_i_folder, exist_ok=True)

clip_index = 0  # 小视频编号

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


for input_video_file in input_video_files:
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))# 获取视频的总帧数
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    if duration < 1:
        print("视频时长小于一秒，跳过处理")
        cap.release()
        continue
    # 初始化变量

    clip_index += 1
    recording = False  # 标记当前是否在记录视频
    out = None  # 视频写入器
    # clip_index = 1  # 小视频编号

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
            box_index = 1
            for box in result.boxes:
                if int(box.cls.item()) == 0:  # 0表示人物

                    # 获取边界框坐标并裁剪图像
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_person = frame[y1:y2, x1:x2]

                    _output_i_folder = os.path.join(output_i_folder, str(clip_index))
                    if not os.path.exists(_output_i_folder):
                        os.makedirs(_output_i_folder)

                    # 保存裁剪结果
                    crop_path = os.path.join(_output_i_folder, f'{str(clip_index).zfill(4)}_c1s1_{frame_index}_{box_index}.jpg')
                    # 128*256
                    cropped_person = cv2.resize(cropped_person, (128, 256))
                    cv2.imwrite(crop_path, cropped_person)

                    person_detected = True
                    box_index += 1
                    break


    # 释放资源
    if recording:
        out.release()  # 如果最后仍在录制，释放写入器

    cap.release()
    cv2.destroyAllWindows()
    print("视频处理完毕，所有小视频已保存！")
