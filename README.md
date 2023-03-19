# Boston housing 
<p>Visando testar uma forma de seleção de features e tuning de hyperparametros irei seguir o seguinte passo a passo para realizar a regressão: 
<li>Carregar os dados e separar em Features e Target
<li>Realizar seleção de features usando o pacote Boruta
<li>Particionar as Features selecionadas e Target em treino e teste
<li>Criar um pipeline com StandardScaler() e LinearRegression()
<li>Realizar tuning atraves de BayesSearchCV do pacote scikit-optimize
<li>Verificar o score do modelo vencedor
<li>Medição de erros (MSE - RMSE - MAE)