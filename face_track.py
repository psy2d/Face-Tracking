import cv2
import sys

major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

if __name__ == '__main__' :

    if len (sys.argv) < 2 :
        print ("Usage: python3 face_track.py < tracker_type > \ntracker_type = BOOSTING | MIL | KCF | TLD | MEDIANFLOW | GOTURN")
        sys.exit (1)

    tracker_type = sys.argv[1].upper()
    cv2.namedWindow("Tracking")

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    multi_tracker = cv2.MultiTracker_create()

    # Read video
    cap = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not cap.isOpened():
        print ("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = cap.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()

    # Detect faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        ok = multi_tracker.add(tracker, frame, (x,y,w,h))
    ok, boxes = tracker.update(frame)

    while True:
        # Read a new frame
        ok, frame = cap.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update multi_tracker
        ok, boxes = multi_tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            count=0
            for newbox in boxes:
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(frame, p1, p2, (200,count*100,count*100),3)
                count+=1
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
