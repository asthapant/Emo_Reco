def get_emotion(location):
    file_emotion = open(location,'r')
    line = file_emotion.readline()
    file_emotion.close()
    return line.strip()[0]


def framelist(main_dir,file_loc):
    if os.path.exists(main_dir)==False:
        print("Not a valid path")
        exit(0)
    file1 = open(file_loc,'w')
    subjects = sorted(os.listdir(main_dir))
    print(subjects)
    i=0
    for subject in subjects:
        sub_loc = main_dir+"/"+subject
        print(sub_loc)
        videos = sorted(os.listdir(sub_loc))
        for video in videos:
            video_loc = sub_loc+"/"+video
            #print(video_loc)
            videos1 = sorted(os.listdir(video_loc))
            for video11 in videos1:
              emotion = get_emotion(video_loc+'/'+video11+"/emotion.txt")
              frames = sorted([int(x[:-4]) for x in os.listdir(video_loc+'/'+video11+"/frames")])
              #for frame in frames:
              file1.write(video_loc+'/'+video11+"/frames/"+","+emotion+'\n')
              i+=1
    print("Total Number of Frames Extracted:",i)
    file1.close()
