from fastapi import FastAPI
from pydantic import BaseModel
import h2o
from h2o.frame import H2OFrame
import pandas as pd
import gradio as gr
from fastapi.responses import RedirectResponse

app = FastAPI()
h2o.init(log_level='ERRR')

model = h2o.load_model('/models/StackedEnsemble_BestOfFamily_1_AutoML_8_20250705_143353')

class LTVIN(BaseModel):
    Location: str
    Income_Level: str
    App_Usage_Frequency: str
    Active_Days: float
    Last_Transaction_Days_Ago: float
    Min_Transaction_Value: float
    Max_Transaction_Value: float
    Loyalty_Points_Earned: float
    Customer_Satisfaction_Score: float
    Issue_Resolution_Time: float


@app.post('/predict')
def predict(input: LTVIN):
    input_dict = input.dict()
    df = pd.DataFrame([input_dict])

    for cat_c in ['Location', 'Income_Level', 'App_Usage_Frequency']:
        if cat_c in df.columns:
            df[cat_c] = df[cat_c].astype('category')

    h2o_df = h2o.H2OFrame(df)
    prediction = model.predict(h2o_df).as_data_frame().values[0][0]
    return f'Prediction of LTV in your case: {float(prediction)}'

def gradio_predict(
    Location,
    Income_Level,
    App_Usage_Frequency,
    Active_Days,
    Last_Transaction_Days_Ago,
    Min_Transaction_Value,
    Max_Transaction_Value,
    Loyalty_Points_Earned,
    Customer_Satisfaction_Score,
    Issue_Resolution_Time
):
    input_dict = {
    "Location": Location,
    "Income_Level": Income_Level,
    "App_Usage_Frequency": App_Usage_Frequency,
    "Active_Days": Active_Days,
    "Last_Transaction_Days_Ago": Last_Transaction_Days_Ago,
    "Min_Transaction_Value": Min_Transaction_Value,
    "Max_Transaction_Value": Max_Transaction_Value,
    "Loyalty_Points_Earned": Loyalty_Points_Earned,
    "Customer_Satisfaction_Score": Customer_Satisfaction_Score,
    "Issue_Resolution_Time": Issue_Resolution_Time,
}  
    df = pd.DataFrame([input_dict])
    h2o_df = h2o.H2OFrame(df)
    prediction = model.predict(h2o_df).as_data_frame().values[0][0]
    return f'Prediction of LTV in your case: {float(prediction)}'

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(choices=["Urban", "Suburban", "Rural"], label='Location'),
        gr.Dropdown(choices=["Low", "Medium", "High"], label='Income_Level'),
        gr.Dropdown(choices=["Rarely", "Sometimes", "Often"], label='App_Usage_Frequency'),
        gr.Number(label='Active_Days'),
        gr.Number(label='Last_Transaction_Days_Ago'),
        gr.Number(label='Min_Transaction_Value'),
        gr.Number(label='Max_Transaction_Value'),
        gr.Number(label='Loyalty_Points_Earned'),
        gr.Number(label='Customer_Satisfaction_Score'),
        gr.Number(label='Issue_Resolution_Time')
    ],
    outputs=gr.Number(label='Predicted LTV'),
    title='FinTech LTV prediction',
    description='Input the data of FinTech company to see estimated Lifetime Value'
)

app.mount('/gradio', iface.app)

@app.get('/')
def root():
    return RedirectResponse(url='/gradio')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)