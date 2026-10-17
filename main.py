from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd

camera = cv2.VideoCapture(0)

emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

times = []
past = {emotion: [] for emotion in emotions}
start = time.time()
frame_counter = 0

last_region = None
last_emotion = ""

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % 5 == 0:
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            scores = result[0]["emotion"]
            last_emotion = result[0]["dominant_emotion"]

            times.append(time.time() - start)

            for em in emotions:
                past[em].append(scores[em])

            r = result[0]["region"]

            if r.get("w", 0) > 0 and r.get("h", 0) > 0:
                last_region = (r["x"], r["y"], r["w"], r["h"])
            else:
                last_region = None

        except Exception as e:
            print("Hata:", e)

    # Çizim her karede yapılır (% 5 bloğunun dışında)
    if last_region is not None:
        x, y, w, h = last_region
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, last_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Emotion Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

# ---- Grafik ----
# ---- Pandas ----
if len(times) > 0:

    df = pd.DataFrame(past)
    df.insert(0, "time", times)

    df.to_csv("emotions.csv", index=False)

    print(df[emotions].describe().round(2))

    smooth = df[emotions].rolling(window=5, min_periods=1).mean()
    smooth.insert(0, "time", df["time"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    smooth.plot(x="time", y=emotions, ax=ax1)
    ax1.set_title(("Emotion History (smoothed)"))
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Score (%)")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    df[emotions].mean().plot(kind="bar", ax=ax2)
    ax2.set_title(("Average Emotion Scores"))
    ax2.set_ylabel(("Average Score (%)"))
    ax2.tick_params(axis="y", rotation=0)



    plt.tight_layout()
    plt.savefig("emotions_graph.png", dpi=150)
    plt.show()
else:
    print("No Emotion History")