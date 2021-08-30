import os 
import cv2
import dlib 
import argparse

from os import walk
from os.path import join


def parser () :
    parser = argparse.ArgumentParser(description="Face detection and crop")
    parser.add_argument("--input","-i",type=str,default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--multi_video", "-m", action='store_true',
                        help="process multiple videos")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try: 
    #if input is webcam 
        return int(video_path)

        
    except ValueError: 
    #if input is video not webcam 

        return video_path


def set_saved_video(input_video, in_filename , output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fps = int(input_video.get(cv2.CAP_PROP_FPS))
    fps = 30
    output_video = output_video +in_filename
    # print("output_video_path : ")
    # print( output_video) 

    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def crop_img(img,x,y,h,w):

    # crop image
    #crop_img = img[y_u:y_d, x_l:x_r]  # notice: first y, then x
    crop_img = img[y:y+h, x:x+w]
    return crop_img


def cut_video (video_path,  in_filename, out_filename ) :

    cap = cv2.VideoCapture(video_path)
    video = set_saved_video(video_path, in_filename, out_filename, (1280,720) )
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #是否為第一幀
    f_frame = False 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(gray)

        for (face) in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            a = face.left()
            b = face.top()
            w = face.right() - a
            h = face.bottom() - b
            if ( f_frame == False ) : #取第一幀的畫面抓取的眶大小
                test = crop_img(frame,a,b,w,h)    
                a1 = a
                b1 = b
                w1 = w
                h1 = h
                f_frame = True  
                print(a1)
                print(b1)
                print(w1)
                print(h1)
            else: #其他幀沿用第一偵抓到的框大小
                try :
                    test = crop_img(frame,a1,b1,w1,h1)
                except Exception as e :
                    print (str(e))

        test = cv2.resize(test,(1280,720))
        video.write(test)
        cv2.imshow("Face Landmarks", test)
        key = cv2.waitKey(1)
        if key == 27:
            break

  
    cap.release()
    video.release()

    #cv2.destroyAllWindows()
            

def main () :

    args = parser() 

    if args.multi_video:
        for root, _, files in walk(args.input):
            print(files)
            for f in files:
                print(f)
                if f.endswith('.mp4'):
                    
                    # print("just for watch : ")
                    #print(os.path.join(root,f))
                    #print(str2int(join(root, f)))
                    cut_video(str2int(join(root, f)),f ,args.out_filename)  
                    #catch multi_video's video_path

if __name__ == '__main__' :
    main()            