from ftplib import FTP
import os





global ft
ftp = FTP('195.69.187.77', user='taxiuser', passwd='K2krJzFBEg9xxsz')
ftp.dir()
lst = ftp.nlst('20211112')
print(lst)
print("connected to FTP")
local_filename = os.path.join('/tmp', 'office.jpg')
out = 'E:\people finding\learnopencv-master\ObjectDetection-YOLO\office.jpg'
with open(out, 'wb') as f:
    ftp.retrbinary('RETR ' + f'{lst[-1]}', f.write)