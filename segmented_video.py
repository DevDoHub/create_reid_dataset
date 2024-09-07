from ultralytics import YOLO
import cv2
import os

"""
检测视频中出现人前5s到人消失后5s，分别保存成一个小视频
遍历两遍，慢
"""

# 加载YOLOv8模型并指定使用GPU
# model = YOLO('yolov8n.pt').to('cuda')  # 将模型加载到GPU上
model = YOLO('yolov8n.pt')  # 将模型加载到CPU上

# 如果没有CUDA支持，可以替换为'to('cpu')'来强制使用CPU

# 输入视频文件路径
video_path = 'test.mp4'  # 替换为你的视频路径
output_folder = 'output_videos'
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数

# 定义变量记录人物出现的时间段
person_appear_frames = []
start_frame = None
end_frame = None

# 遍历视频的每一帧
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行检测
    results = model(frame)  # 此处会使用CPU/GPU进行推理
    person_detected = False

    # 检查是否检测到人物
    for result in results:
        for box in result.boxes:
            if int(box.cls.item()) == 0:  # 0表示人物
                person_detected = True
                break

    # 如果检测到人物
    if person_detected:
        if start_frame is None:  # 第一次检测到人物
            start_frame = i
        end_frame = i  # 更新结束帧为当前帧
    else:
        # 如果没有检测到人物且之前有检测到，记录时间段
        if start_frame is not None and end_frame is not None:
            person_appear_frames.append((start_frame, end_frame))
            start_frame = None
            end_frame = None

# 处理视频尾部可能存在的未保存段
if start_frame is not None and end_frame is not None:
    person_appear_frames.append((start_frame, end_frame))

cap.release()  # 释放视频资源

# 创建并保存小视频片段
cap = cv2.VideoCapture(video_path)
for idx, (start, end) in enumerate(person_appear_frames):
    # 计算前5秒和后5秒的帧数
    start_cut = max(0, start - int(5 * fps))
    end_cut = min(frame_count - 1, end + int(5 * fps))

    # 设置视频写入器
    output_video_path = os.path.join(output_folder, f'person_clip_{idx + 1}.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_cut)  # 从剪辑起始帧开始

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码格式
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 写入剪辑视频片段
    for i in range(start_cut, end_cut + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    print(f"保存小视频: {output_video_path}")

cap.release()
cv2.destroyAllWindows()
print("所有小视频片段已保存完毕！")
