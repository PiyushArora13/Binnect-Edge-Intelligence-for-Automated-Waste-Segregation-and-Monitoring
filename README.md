**Binnect** is a distributed embedded system integrating **Edge Artificial Intelligence (AI)** and **Internet of Things (IoT)** for real-time waste classification and bin-level monitoring. A **TensorFlow Lite (TFLite) image classification model** is deployed on a **Raspberry Pi 5** for on-device inference, while an **ESP8266-based module** performs ultrasonic-based fill-level detection and hosts a lightweight web dashboard.


## **🏗️ System Architecture**

The system follows a **distributed architecture** consisting of two primary nodes:

### **1. Edge Inference Node (Raspberry Pi 5)**

* Captures images using Pi Camera
* Performs real-time inference using **TFLite interpreter**
* Classifies waste into **Wet / Dry**
* Controls **GPIO-based LED indicators**

### **2. IoT Node (ESP8266)**

* Measures bin levels using **ultrasonic sensors**
* Hosts a **web dashboard (HTML, CSS)**
* Implements **threshold-based bin status detection**
* Communicates bin availability to Raspberry Pi


## **🔄 Data Flow Pipeline**

1. **Proximity Detection**

   * Object detected using an ultrasonic sensor

2. **Image Acquisition**

   * Frame captured via Pi Camera

3. **Preprocessing**

   * Resize to **224×224**
   * Format conversion to model input tensor

4. **Inference**

   * Forward pass using **TFLite interpreter**
   * Output probability vector generated

5. **Decision Logic**

   * `argmax()` used for class selection
   * Output mapped to **Wet / Dry**

6. **Conditional Actuation**

   * If **bin available** → LED glows for 2 seconds
   * If **bin full** → actuation blocked

7. **Monitoring**

   * ESP8266 updates real-time dashboard


## **🧠 Machine Learning Details**

* **Model Type:** Convolutional Neural Network (CNN)
* **Framework:** TensorFlow Lite
* **Technique:** Transfer Learning
* **Deployment:** Edge Inference
* **Input Shape:** (1, 224, 224, 3)
* **Data Type:** uint8 (Quantized Model)

### **Preprocessing Pipeline**

* Resize → Normalize → Expand Dimensions
* Channel correction (BGRA → BGR if required)


## **⚙️ Embedded System Design**

### **Raspberry Pi 5**

* Linux-based system
* Executes Python-based inference pipeline
* Interfaces:

  * **CSI Camera**
  * **GPIO (LED Control)**
  * **Serial / Network Communication**

### **ESP8266 (NodeMCU)**

* Programmed using Arduino framework
* Interfaces:

  * **Ultrasonic Sensor (HC-SR04)** 
* Functionalities:

  * Fill-level monitoring
  * Dashboard hosting
  * Threshold-based decision logic


## **🔌 Inter-Device Communication**

Communication between Raspberry Pi and ESP8266 can be implemented using:

* **UART (Serial Communication)** *(recommended)*
* **HTTP-based communication (REST APIs)**
* **GPIO-based signaling (basic implementation)**

## **🌐 Web Dashboard**

A lightweight dashboard hosted on ESP8266.

### **Technologies Used**

* **HTML**
* **CSS**


### **Features**

* Real-time bin level display
* Status indicators:

  * **AVAILABLE**
  * **FULL**
* Low-latency response


## **📊 Control Logic**

| **Condition**                   | **Action**                      |
| ------------------------------- | ------------------------------- |
| Object detected & bin available | Classification + LED indication |
| Object detected & bin full      | Block LED + show alert          |
| No object detected              | Idle state                      |



## **🛠️ Dependencies**

### **Raspberry Pi 5**

* numpy
* opencv-python
* tflite-runtime
* picamera2
* RPi.GPIO

### **ESP8266**

* ESP8266WiFi
* ESPAsyncWebServer 

## **🚀 Future Enhancements**

* Servo-based lid automation
* Cloud integration (**AWS IoT / MQTT**)
* Mobile application interface
* Data logging and analytics
* Multi-class waste classification


## **🧩 Key Contributions**

* Real-time **Edge AI inference system**
* Integration of **Computer Vision + IoT + Embedded Systems**
* Scalable and modular architecture
* Low-cost implementation

