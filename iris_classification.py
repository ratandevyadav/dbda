import streamlit as st
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# Train the SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Define the app
def app():
    # Set the app title
    st.title("Iris Flower Classification App Developed")

    # Define the sidebar with input options
    st.sidebar.header('Input Features')
    sepal_length = st.sidebar.slider('Sepal length', 0.0, 10.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal width', 0.0, 10.0, 5.0)
    petal_length = st.sidebar.slider('Petal length', 0.0, 10.0, 5.0)
    petal_width = st.sidebar.slider('Petal width', 0.0, 10.0, 5.0)

    # Show the input values
    st.write('Input values:')
    st.write('Sepal length:', sepal_length)
    st.write('Sepal width:', sepal_width)
    st.write('Petal length:', petal_length)
    st.write('Petal width:', petal_width)

    # Make predictions using the model
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = iris.target_names[prediction[0]]

    # Show the predicted species
    st.write('\n')
    st.write('Predicted species:')
    st.write(species)

    # Evaluate the accuracy of the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write('\n')
    st.write('Accuracy:', accuracy)

# Run the app
if __name__ == '__main__':
    app()
