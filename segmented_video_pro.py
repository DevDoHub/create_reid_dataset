from ultralytics import YOLO
import cv2
import os


"""
检测视频中出现人到人消失，分别保存成一个小视频
遍历一遍, 保存
"""

# 加载YOLOv8模型
# model = YOLO('yolov8n.pt').to('cuda')  # 将模型加载到GPU上
model = YOLO('yolov8n.pt')  # 请使用适合的模型路径

# 输入视频文件路径
video_path = 'test.mp4'  # 替换为你的输入视频路径
output_folder = 'output_videos1'
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
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

                # # 获取边界框坐标并裁剪图像
                # x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # cropped_person = frame[y1:y2, x1:x2]

                # # 保存裁剪结果
                # crop_path = os.path.join(output_folder, f'person_标签(视频段){clip_index}_{x1}_{y1}.jpg')
                # # 128*256
                # cropped_person = cv2.resize(cropped_person, (256, 128))
                # cv2.imwrite(crop_path, cropped_person)

                person_detected = True
                break

    # 判断是否需要开始或停止录制
    if person_detected:
        if not recording:  # 如果当前没有录制并检测到人物，开始录制
            output_video_path = os.path.join(output_folder, f'person_clip_{clip_index}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            recording = True
            clip_index += 1
            print(f"开始录制小视频: {output_video_path}")

        # 写入当前帧
        out.write(frame)
    else:
        if recording:  # 如果当前正在录制但没有检测到人物，停止录制
            out.release()
            recording = False
            print("停止录制并保存小视频")

    frame_index += 1

# 释放资源
if recording:
    out.release()  # 如果最后仍在录制，释放写入器

cap.release()
cv2.destroyAllWindows()
print("视频处理完毕，所有小视频已保存！")
