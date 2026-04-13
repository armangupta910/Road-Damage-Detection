import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection App",
    page_icon="🚧",
)

st.title("🚧 Enhanced Road Damage Detection")
st.subheader("With Severity Estimation and Intelligent Risk Scoring")
st.divider()

st.markdown(
    """
    This application extends traditional road damage detection by introducing **quantitative severity estimation**
    and an **intelligent risk scoring mechanism**.

    Instead of only detecting damages, the system analyzes:
    - **Spatial extent** of damage
    - **Detection confidence**
    - **Distribution patterns**

    to generate a **real-time road condition assessment**.
    """
)

st.divider()

st.markdown("### Detectable Damage Types")
st.markdown(
    """
    The model is trained on the **RDD2022 (India subset)** dataset using **YOLOv8 Small** and can detect four types of road damage:

    | Damage Type         | Class Weight |
    |---------------------|--------------|
    | Longitudinal Crack  | 0.5          |
    | Transverse Crack    | 0.6          |
    | Alligator Crack     | 0.8          |
    | Potholes            | 1.0          |
    """
)

st.divider()

st.markdown("### Core Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **Severity**  
        Pixel-level union of bounding boxes divided by total image area.  
        *No double counting.*
        """
    )

with col2:
    st.info(
        """
        **Weighted Severity**  
        Confidence × Class Weight per pixel, normalized by image area.
        """
    )

with col3:
    st.info(
        """
        **Risk Score**  
        Multi-factor score combining Severity, Weighted Severity, Count, Density & Spread.
        """
    )

st.divider()

st.markdown("###Road Condition Index (RCI)")
st.markdown(
    """
    ```
    RCI = (1 - Weighted Severity) × 100
    ```

    | RCI Range | Condition |
    |-----------|-----------|
    | 80 – 100  | ✅ Good   |
    | 50 – 80   | ⚠️ Moderate |
    | < 50      | 🔴 Severe |
    """
)

st.divider()

st.markdown("### Detection Pipeline")
st.markdown(
    """
    ```
    Image → YOLOv8 → Bounding Boxes → Pixel Union Mask
         → Severity → Density + Spread → Risk Score → RCI
    ```
    """
)

st.divider()

st.markdown("### 🖥️ How to Use")
st.markdown(
    """
    Each mode outputs:
    - Bounding boxes on detected damage
    - Severity score & Weighted Severity
    - Risk Score & RCI value
    - Class-wise breakdown chart
    - Option to download the prediction image
    """
)

st.divider()

st.markdown(
    """
    #### 📎 Links & References
    - Dataset: [CRDDC2022 – RDD2022](https://github.com/sekilab/RoadDamageDetector)
    - Model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
    - Framework: [Streamlit](https://streamlit.io/)

    """
)