import cv2
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def detect_cone_and_display(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([5, 150, 150])         # hsv values to be  improved
    upper_orange = np.array([15, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "CONE DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        resized_image = cv2.resize(image, (0, 0), fx = 0.4, fy = 0.4)
        
        cv2.imshow("Cone Detection", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return h
    else:
        raise ValueError("No cone detected in the image.")

def fit_distance_height_relation(distances, heights):
    def inverse_func(distance, a, b):
        return a / distance + b

    params, _ = curve_fit(inverse_func, distances, heights)
    return params

def predict_distance(cone_height, params):
    a, b = params
    return a / (cone_height - b)

if __name__ == "__main__":
    image_paths = ["cone_15.jpeg", "cone_60.jpeg", "cone_120.jpeg"]
    distances = np.array([15, 60, 120])  

    heights = np.array([detect_cone_and_display(path) for path in image_paths])
    print(f"Detected cone heights (in pixels): {heights}")

    params = fit_distance_height_relation(distances, heights)
    print(f"Fitted parameters: a = {params[0]:.2f}, b = {params[1]:.2f}")

    new_image_path = "cone_predict.jpeg"
    new_cone_height = detect_cone_and_display(new_image_path)
    predicted_distance = predict_distance(new_cone_height, params)
    print(f"Predicted distance: {predicted_distance:.2f} cm")
    smooth_distances = np.linspace(10, 130, 100)
    fitted_heights = params[0] / smooth_distances + params[1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(distances, heights, color='red', label='Measured Heights')
    plt.plot(smooth_distances, fitted_heights, label=f'Fit: a={params[0]:.2f}, b={params[1]:.2f}', color='blue')
    plt.xlabel('Distance (cm)')
    plt.ylabel('Cone Height in Pixels')
    plt.title('Cone Height vs. Distance')
    plt.legend()
    plt.grid(True)

    
    plt.show()
