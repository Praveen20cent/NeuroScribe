# NeuroScribe
Translating Thoughts into Written Words with Robotic Finesse and Vocal Feedback

This project presents the development of a handwriting assistance system designed for individuals with limited motor function, leveraging Brain-Computer Interface (BCI) technology, advanced machine learning algorithms, and Internet of Things (IoT) devices. The system utilizes the OpenBCI Ultracortex Mark IV EEG headset to capture brain signals corresponding to the user's intent to write specific letters. Signal preprocessing involves bandpass filtering to confine signals within the 0-60 Hz range, followed by Variational Mode Decomposition (VMD) to decompose signals, reduce noise, and eliminate irrelevant frequencies.

The preprocessed EEG data are fed into a deep learning model trained to classify EEG patterns associated with individual letters. An IoT-enabled device with a Wi-Fi module facilitates real-time control of a CNC plotter, which physically reproduces the user's intended letter. Additionally, an integrated voice speaker provides immediate auditory feedback by announcing the predicted letter, enhancing user interaction and control.

This project advances assistive technologies by introducing a practical and intuitive BCI-controlled robotic handwriting system. Future enhancements may include adaptive learning algorithms for improved prediction accuracy, personalized calibration for tailored performance, and exploration of alternative input modalities to extend the system's applicability beyond handwriting. Further improvements, such as sophisticated pattern recognition algorithms and dynamic time warping for signal alignment, are also proposed to enhance the system's accuracy, speed, efficiency and user-friendliness.


## FLOW CHART
![Work Flow](https://github.com/user-attachments/assets/14804a49-4fb9-4a75-b373-08e8467633c7)


Computational Efficiency

When employing Variational Mode Decomposition (VMD) for processing a single set of signals (e.g., set 'A'), the time taken was 5 minutes and 9 seconds on both cloud and local systems. In contrast, using Empirical Mode Decomposition (EMD) reduced this processing time to under 40 seconds per signal set. This demonstrates EMD's superiority in both computational and time efficiency compared to VMD.


Machine Learning Model Performance

For machine learning models, EMD significantly improved classification accuracy:

Support Vector Machine (SVM):
Accuracy increased from 90% (with VMD) to 97% (with EMD).

Random Forest:
Accuracy improved from 99% (with VMD) to 100% (with EMD).

This performance boost can be attributed to EMD’s ability to decompose signals into intrinsic mode functions (IMFs) that capture more granular features of the signal.


Deep Learning Model Performance

Initial implementation of EMD for deep learning faced challenges due to differences in the size of processed data compared to VMD. These issues were resolved by reshaping and preprocessing the data appropriately. Both methods achieved an accuracy of 100% when applied to deep learning models.

Further optimizations revealed that EMD performed best with the following optimizers:
Nadam: 100% accuracy achieved in 15 epochs.
Adam: 100% accuracy achieved in 15 epochs.
RMSprop: 100% accuracy achieved in 10 epochs.


Workflow Adjustments for Machine Learning

The feature extraction process for machine learning models underwent significant changes. Instead of relying on features like Katz centrality, PSD, and spectral entropy, the following statistical features were utilized:

Mean
Variance
Entropy
Energy


The workflow remained consistent:

EMD was applied to decompose the signals.
Statistical features were extracted using a custom function.
The dataset was split into training and testing sets.


Workflow Adjustments for Deep Learning

Due to the dimensional changes introduced by EMD (an increase in dimension by 1), direct use of EMD data in the recurrence plot function was not feasible. The updated process involved:
Extracting features from EMD using a specialized function.
Reshaping the extracted features to match the input requirements of the recurrence plot function.
Proceeding with the same workflow as in the original VMD-based implementation.

Conclusion

The integration of EMD into ECG classification workflows has proven to be computationally efficient and effective in improving model performance. By tailoring feature extraction and optimizing workflows for both machine learning and deep learning, EMD not only outperformed VMD in certain metrics but also maintained comparable accuracy where VMD excelled. The adjustments made ensure the robustness and adaptability of the classification system.
