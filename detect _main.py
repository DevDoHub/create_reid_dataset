import uuid
from ultralytics import YOLO
import cv2
import os

"""
检测视频中的人物并裁剪保存

"""

# 加载预训练的YOLOv8模型
# model = YOLO('yolov8n.pt').to('cuda')  # 将模型加载到GPU上
model = YOLO('yolov8n.pt')  # 选择适当的YOLOv8模型，如yolov8n.pt、yolov8s.pt等

# 设置视频路径和输出文件夹
video_path = 'output_videos1'

# 判断video_path是视频文件还是目录
if not os.path.isdir(video_path):
    video_files = [video_path]
else:
    # 获取目录下所有的视频文件路径
    video_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.mp4')]

for video_file in video_files:
    output_folder = 'output_' + os.path.splitext(video_file)[0]

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_file)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLOv8模型进行检测
        results = model(frame)

        # 遍历每一个检测到的对象
        for result in results:
            for box in result.boxes:  # 获取每个检测框
                cls = int(box.cls.item())  # 类别ID
                if cls == 0:  # 检测到的类别为0表示"person"
                    # 获取边界框坐标并裁剪图像
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cropped_person = frame[y1:y2, x1:x2]

                    # 保存裁剪结果
                    # video_files.index(video_file)表示当前视频文件的索引, 相当于标签
                    # frame_count表示当前视频的帧数
                    crop_path = os.path.join(output_folder, f'person_{frame_count}_{video_files.index(video_file)}_{x1}_{y1}.jpg')
                    # 128*256, 等比例缩放
                    
                    cropped_person = cv2.resize(cropped_person, (128, 256))
                    cv2.imwrite(crop_path, cropped_person)

        frame_count += 1

    # 释放视频资源
    cap.release()
    cv2.destroyAllWindows()

    print("人物裁剪完成，图像已保存到:", output_folder)




# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()