import cv2
from ultralytics import YOLO
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # ⭐ for visualization


def detect_traffic(source):
    model_n = YOLO('yolov8n.pt')  # YOLOv8n model
    model_s = YOLO('yolov8s.pt')  # YOLOv8s model
    class_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}  # ⭐ store detected class counts
    density_pairs = []  # List to store (density_n, density_s) tuples

    # If source is an image file
    if isinstance(source, str) and os.path.splitext(source)[1].lower() in ['.jpg', '.jpeg', '.png']:
        frame = cv2.imread(source)
        if frame is None:
            print("❌ Could not load image.")
            return

        # Run detection with both models
        results_n = model_n(frame)
        results_s = model_s(frame)

        # Count vehicles for model_n
        cars_n = 0
        for result in results_n:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model_n.names[cls]
                if name in class_counts:
                    class_counts[name] += 1
                    cars_n += 1

        # Count vehicles for model_s
        cars_s = 0
        for result in results_s:
            for box in result.boxes:
                cls = int(box.cls[0])
                name = model_s.names[cls]
                if name in class_counts:
                    cars_s += 1  # Note: class_counts is shared, but for density we use separate counts

        # Classify density for model_n
        if cars_n < 5:
            density_n = "LOW"
            color_n = (0, 255, 0)
        elif cars_n < 15:
            density_n = "MEDIUM"
            color_n = (0, 255, 255)
        else:
            density_n = "HIGH"
            color_n = (0, 0, 255)

        # Classify density for model_s
        if cars_s < 5:
            density_s = "LOW"
        elif cars_s < 15:
            density_s = "MEDIUM"
        else:
            density_s = "HIGH"

        # Store density pair
        density_pairs.append((density_n, density_s))

        # Display using model_n results
        annotated = results_n[0].plot()
        cv2.putText(annotated, f"Traffic Density (YOLOv8n): {density_n} ({cars_n} vehicles)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_n, 3)
        cv2.imshow("🖼️ Traffic Density (Image)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Otherwise, treat as video or webcam
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open source {source}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ End of stream or cannot read frame.")
                break

            # Run detection with both models
            results_n = model_n(frame)
            results_s = model_s(frame)

            # Count vehicles for model_n
            cars_n = 0
            for result in results_n:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = model_n.names[cls]
                    if name in class_counts:
                        class_counts[name] += 1
                        cars_n += 1

            # Count vehicles for model_s
            cars_s = 0
            for result in results_s:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    name = model_s.names[cls]
                    if name in class_counts:
                        cars_s += 1

            # Classify density for model_n
            if cars_n < 5:
                density_n = "LOW"
                color_n = (0, 255, 0)
            elif cars_n < 15:
                density_n = "MEDIUM"
                color_n = (0, 255, 255)
            else:
                density_n = "HIGH"
                color_n = (0, 0, 255)

            # Classify density for model_s
            if cars_s < 5:
                density_s = "LOW"
            elif cars_s < 15:
                density_s = "MEDIUM"
            else:
                density_s = "HIGH"

            # Store density pair
            density_pairs.append((density_n, density_s))

            # Display using model_n results
            annotated = results_n[0].plot()
            cv2.putText(annotated, f"YOLOv8n: {density_n} ({cars_n} vehicles) | YOLOv8s: {density_s} ({cars_s} vehicles)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_n, 2)

            cv2.imshow("🚦 Traffic Density Detection (Model Comparison)", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ⭐ After detection → Display confusion matrix for density predictions
    print("\nDetected Vehicle Counts:", class_counts)
    print(f"Total frames processed: {len(density_pairs)}")

    if density_pairs:
        # Extract predictions
        y_true = [pair[0] for pair in density_pairs]  # YOLOv8n predictions (as "true" for comparison)
        y_pred = [pair[1] for pair in density_pairs]  # YOLOv8s predictions

        # Map labels to indices
        label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        y_true_idx = [label_map[label] for label in y_true]
        y_pred_idx = [label_map[label] for label in y_pred]

        # Compute confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LOW", "MEDIUM", "HIGH"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix: YOLOv8n vs YOLOv8s Density Predictions\n(Rows: YOLOv8n, Columns: YOLOv8s)")
        plt.show()

        # Calculate agreement percentage
        total_agreements = np.trace(cm)  # Sum of diagonal
        total_predictions = np.sum(cm)
        agreement_percentage = (total_agreements / total_predictions) * 100 if total_predictions > 0 else 0

        print(f"\nModel Comparison Summary:")
        print(f"Total frames: {total_predictions}")
        print(f"Agreements: {total_agreements}")
        print(f"Agreement Percentage: {agreement_percentage:.2f}%")

        # Print per-class agreement rate
        for i, label in enumerate(["LOW", "MEDIUM", "HIGH"]):
            row_sum = np.sum(cm[i, :])
            if row_sum > 0:
                agreement_rate = (cm[i, i] / row_sum) * 100
                print(f"Agreement rate for {label}: {agreement_rate:.2f}% ({cm[i, i]}/{row_sum} times YOLOv8n and YOLOv8s both predicted {label})")
    else:
        print("No density pairs collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Density Estimation (Image / Video / Live)")
    parser.add_argument("--source", type=str, default="0",
                        help="Path to image/video file, or '0' for webcam")
    args = parser.parse_args()

    source = int(args.source) if args.source == "0" else args.source
    detect_traffic(source)
