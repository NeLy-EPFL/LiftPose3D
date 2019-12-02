import opencv as cv2

data_dirs = ["/mnt/NAS/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/side_view_191125_PR_Fly1_001_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/002_prism/behData/images/side_view_191125_PR_Fly1_002_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/003_prism/behData/images/side_view_191125_PR_Fly1_003_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/004_prism/behData/images/side_view_191125_PR_Fly1_004_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/top_view_191125_PR_Fly1_001_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/002_prism/behData/images/top_view_191125_PR_Fly1_002_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/003_prism/behData/images/top_view_191125_PR_Fly1_003_prism/",
             "/mnt/NAS/SG/prism_data/191125_PR/Fly1/004_prism/behData/images/top_view_191125_PR_Fly1_004_prism/"]

for dd in data_dirs:
    video_name = dd.split("/")
    video_name = video_name[len(video_name)-2]
    print(video_name)


    vw = cv2.VideoWriter(dd+video_name+".mp4", 0, 0, )
