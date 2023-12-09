# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:07:07 2023

@author: USER
"""

import streamlit as st
from io import StringIO
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings 
import streamlit as st
from function import load_data1

# Define CSS styles


def app():
    css = """
        <style>
        .tooltip {
            position: absolute;
            top: -5px;
            left: 0px;
            z-index: 5;
            margin-top: 0px;
            padding-top: 10px;
            display: inline-block;
            cursor: help;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #f9f9f9;
            color: black;
            text-align: center;
            border-radius: 4px;
            padding: 5px;
            position: absolute;
            z-index: 10;
            bottom: -125%;
            left: 25px;
            transform: translateX(50%);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)

    st.title("Make your own decission Tree Classifier")

    st.markdown(
        '<div class="tooltip">Alpha for Pruning ℹ️<span class="tooltiptext">Pruning is a way to reduces the size of decision trees by removing sections of the tree that are non-critical and redundant to avoid over-fitting.<br><br>Non-negative float, default=0.00</span></div>',
        unsafe_allow_html=True,
    )
    ccp_alpha1 = st.number_input(" ", min_value=0.0)

    st.markdown(
        '<div class="tooltip">Class Weight ℹ️<span class="tooltiptext">Determine class weights. The None value indicates that all classes have the same weight. Balanced value is one approach to handle class imbalance in a decision tree. The purpose of a balanced score is to ensure that the formation of a decision tree does not tend to favor the dominant majority class.<br><br>{“None”, “balanced”}, default=”None” </span></div>',
        unsafe_allow_html=True,
    )
    class_weight1 = st.selectbox(" ", (None, "balanced"), key="option1")
    # class_weight1 = st.selectbox(' ', (None, 0.5, 1, 2), key='option1') # iki punya vincent

    st.markdown(
        '<div class="tooltip">Criterion ℹ️<span class="tooltiptext">The function to measure the quality of a split.<br><br>{“gini”, “entropy”}, default=”gini”</span></div>',
        unsafe_allow_html=True,
    )
    criterion1 = st.selectbox(" ", ("gini", "entropy"), key="option2")
    # criterion1 = st.selectbox(' ', ('entropy', 'gini', 'log_loss'), key='option2') # iki punya vincent

    st.markdown(
        '<div class="tooltip">Max Depth ℹ️<span class="tooltiptext">The maximum depth of the tree.<br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_depth1 = st.selectbox(" ", (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), key="option3")

    st.markdown(
        '<div class="tooltip">Max Features ℹ️<span class="tooltiptext">The maximum feature that will be used for learning. If you select the number, that makes machine decide the best feature for learning based on number of features you choose. <br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_features1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option4")

    st.markdown(
        '<div class="tooltip">Max Leaf Nodes ℹ️<span class="tooltiptext">The maximum number of leaf nodes to be built in a decision tree. <br><br>integer, default=None</span></div>',
        unsafe_allow_html=True,
    )
    max_leaf_nodes1 = st.selectbox(" ", (None, 1, 2, 3, 4), key="option5")

    st.markdown(
        '<div class="tooltip">Max Impurity Decrease ℹ️<span class="tooltiptext">Parameters used to limit the separation at each node based on the desired reduction in impurity. If the impurity decrease exceeds the threshold value specified by Max Impurity Decrease, separation will be performed. However, if the decrease in impurity does not reach the threshold value, then the node will be considered as a terminal node (leaf node) and will not be divided further. Higher values ​​will result in a shallower decision tree with fewer splits, while lower values ​​can result in a deeper decision tree with more splits. <br><br>Non-Negative Float, default=0.0</span></div>',
        unsafe_allow_html=True,
    )
    min_impurity_decrease1 = st.selectbox(
        " ", (0.0, 0.1, 0.01, 0.2, 0.02), key="option6"
    )

    st.markdown(
        '<div class="tooltip">Min Samples Leaf ℹ️<span class="tooltiptext">Determines the minimum number of samples that must be in a leaf node. If the number of samples is below this, the node will not be subdivided. <br><br>integer, default=1</span></div>',
        unsafe_allow_html=True,
    )
    min_samples_leaf1 = st.selectbox(" ", (1, 2), key="option7")

    st.markdown(
        '<div class="tooltip">Min Samples Split ℹ️<span class="tooltiptext">Determine the minimum number of samples needed to perform a knot split. If the number of samples is below this number, no separation will be performed. <br><br>integer, default=1</span></div>',
        unsafe_allow_html=True,
    )
    min_samples_split1 = st.selectbox(" ", (2, 3, 4), key="option8")

    st.markdown(
        '<div class="tooltip">Min Weight Fraction Leaf ℹ️<span class="tooltiptext">Parameter that controls the minimum sample weight fraction required for a node to become a leaf node (terminal node) in a decision tree. By setting a higher value, the decision tree tends to be simpler with fewer divisions and fewer terminal nodes. Conversely, with lower values, the decision tree tends to be more complex with more divisions and more terminal nodes. <br><br>Non-Negative Float, default=0.0</span></div>',
        unsafe_allow_html=True,
    )
    min_weight_fraction_leaf1 = st.selectbox(" ", (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

    st.markdown(
        '<div class="tooltip">Splitter ℹ️<span class="tooltiptext">Determine the best separation strategy to use. If using <b>"best"<b>, for all features, the algorithm selects the "best" point to split, then choose the best feature as the final decision. If using <b>"random"<b>, for all features, the algorithm "randomly" selects a point to split, then choose the best feature as the final decision.<br><br>{“best”, “random”}, default="best"</span></div>',
        unsafe_allow_html=True,
    )
    splitter1 = st.selectbox(" ", ("best", "random"), key="option9")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        # string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        
        st.header("Your Data")
        data = dataframe
        st.dataframe(data)
            
        X = dataframe.iloc[:, :-1]  # Select all columns except the last one as input features
        y = dataframe.iloc[:, -1]   # Select the last column as the target column

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, stratify=y
        )

        treeClass = tree.DecisionTreeClassifier(
            ccp_alpha=ccp_alpha1,
            class_weight=class_weight1,
            criterion=criterion1,
            max_depth=max_depth1,
            max_features=max_features1,
            max_leaf_nodes=max_leaf_nodes1,
            min_impurity_decrease=min_impurity_decrease1,
            min_samples_leaf=min_samples_leaf1,
            min_samples_split=min_samples_split1,
            min_weight_fraction_leaf=min_weight_fraction_leaf1,
            splitter=splitter1,
        )

        treeClass.fit(X_train, y_train)
        y_pred = treeClass.predict(X_test)
        treeAccuracy = accuracy_score(y_pred, y_test)
        # joblib.dump(treeAccuracy, "model.sav")
        st.text("Accuracy From This Model: " + str(treeAccuracy * 100) + "%")

        st.text("")
        st.text("")
        st.header("Decission Tree Model for this dataset")
        dot_data = tree.export_graphviz(
            decision_tree=treeClass,
            max_depth=5,
            out_file=None,
            filled=True,
            rounded=True,
            feature_names=[str(i) for i in range(X_train.shape[1])],
            class_names=y.unique().astype(str),
        )

        st.graphviz_chart(dot_data)

        st.text("")
        st.text("")
        st.header("Confusion Matrix from testing in this dataset")
        cm = confusion_matrix(y_test, y_pred)
        labels = y.unique().astype(str)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal", colorbar=False)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        st.pyplot(fig)

        # if st.checkbox("Prediction Using This Model"):
        #     warnings.filterwarnings('ignore')
        #     st.text("")
        #     # pembagian kolom
        #     col1, col2, col3 = st.columns(3)
            
        #     with col1:
        #         selected_option = st.selectbox("Island: ", ["Torgersen", "Biscoe", "Dream"])
        #         # selected_option = st.radio("Island: ", ("Torgersen", "Biscoe", "Dream"))
        #         if selected_option == "Torgersen":
        #             island = 0.0
        #         elif selected_option == "Biscoe":
        #             island = 1.0
        #         else:
        #             island = 2.0
        #         # island = st.text_input('Masukkan Pulau Asal')
        #     with col1:
        #         culmen_length_mm = st.text_input('Culmen Length (mm)')
        #     with col2:
        #         culmen_depth_mm = st.text_input('Culmen Depth (mm)')
        #     with col2:
        #         flipper_length_mm = st.text_input('Flipper Length (mm)')
        #     with col3:
        #         body_mass_g = st.text_input('Body Mass (g)')
        #     with col3:
        #         selected_option = st.selectbox("Sex: ", ["Male", "Female"])
        #         if selected_option == "Male":
        #             sex = 0.0
        #         elif selected_option == "Female":
        #             sex = 1.0
            
        #     data = {'island': island,
        #             'culmen_length_mm': culmen_length_mm,
        #             'culmen_depth_mm': culmen_depth_mm,
        #             'flipper_length_mm': flipper_length_mm,
        #             'body_mass_g': body_mass_g,
        #             'sex': sex
        #             }
        #     fitur = pd.DataFrame(data, index=[0])
            
        #     data = load_data1()
        #     # st.dataframe(data)
        #     data["island"] = data["island"].replace({"Torgersen": 0, "Biscoe": 1, "Dream": 2})
        #     data["sex"] = data["sex"].replace({"MALE": 0, "FEMALE": 1})
        #     x_data = data[["island", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]]
        #     y_target = data["species"]
        #     X_train, X_test, y_train, y_test = train_test_split(
        #         x_data, y_target, test_size=0.33, stratify=y_target
        #     )
        #     treeClass = tree.DecisionTreeClassifier(
        #         ccp_alpha=ccp_alpha1,
        #         class_weight=class_weight1,
        #         criterion=criterion1,
        #         max_depth=max_depth1,
        #         max_features=max_features1,
        #         max_leaf_nodes=max_leaf_nodes1,
        #         min_impurity_decrease=min_impurity_decrease1,
        #         min_samples_leaf=min_samples_leaf1,
        #         min_samples_split=min_samples_split1,
        #         min_weight_fraction_leaf=min_weight_fraction_leaf1,
        #         splitter=splitter1,
        #     )
        #     treeClass.fit(X_train, y_train)
        #     if st.button("Predict"):
        #         y_pred = treeClass.predict(fitur)
        #         st.success("Species Penguin: " + y_pred)