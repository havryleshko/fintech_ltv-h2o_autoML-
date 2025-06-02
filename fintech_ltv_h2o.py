import h2o
from h2o.automl import H2OAutoML


h2o.init(max_mem_size='4G')

df = h2o.import_file('/Users/ohavryleshko/Documents/GitHub/AutoML/FinTechLTV/digital_wallet_ltv_dataset.csv')
#inspecting loaded data
print(df.head(10))
print('Describing the data...')
print(df.describe())

