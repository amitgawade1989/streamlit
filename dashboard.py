import streamlit as st
import linear_regression as lr

st.title("Linear Regression")

years_of_experience = st.number_input('Years of Experience', 0, 10, step=1,)
# print(years_of_experience)

predicted_salary = lr.predicted_salary(years_of_experience)

st.subheader(f"Predicted Salary for years of experience {years_of_experience} is Rs.{predicted_salary[0]}")

lr.display_graph()