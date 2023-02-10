import numpy as np

class Adaline:
	
	def __init__(self,taxa_aprendizado = 1e-2, max_epochs = 10000, limiar_ativacao = -1,tolerancia = 1e-5):
		
		self.taxa_aprendizado = taxa_aprendizado
		self.max_epochs = max_epochs
		self.tolerancia = tolerancia
		self.qt_cols = 0
		self.qt_rows = 0
		self.limiar_ativacao = 0
		self.pesos = np.array([])
		self.limiar_ativacao = limiar_ativacao
		self.histpeso = {}
		self.p = 0
		self.histerr = {}

	
	def FuncaoAtivacao(self, sum: float)-> int:
		'''Função de ativação degrau bipolar'''

		return 1 if sum > 0 else -1


	def fit(self,X,label) -> bool:
		'''Treinando a rede neural'''

		self.qt_cols = X.shape[1]
		self.qt_rows = X.shape[0]
		total = X.size

		self.p = self.qt_rows


		self.pesos = np.random.rand(self.qt_cols)


		limiar = np.array([self.limiar_ativacao])
		self.pesos = np.hstack([limiar,self.pesos])

		X = np.hstack([-1*np.ones(self.qt_rows).reshape(-1,1),X])

		#Iniciando contador de epoca
		epoch = 0

		#erros
		erro_agora, erro_anterior = 1, 0

		# Iniciando o processo de treinamento
		while(epoch < self.max_epochs and np.abs(erro_agora - erro_anterior) > self.tolerancia):
			

			
			erro_anterior = erro_agora
			err = 0
			vetor_erro_acumulado = np.zeros(self.qt_cols+1)
			#Ponderando a entrada com os pesos
			for i in range(self.qt_rows):
				# Realizando a soma ponderada
				sum_ = (X[i]*self.pesos).sum()

				# atualizando os pesos
				#self.pesos = self.pesos + (self.taxa_aprendizado * (label[i] - sum_) * X[i])
				
				vetor_erro_acumulado = vetor_erro_acumulado + ((label[i] - sum_)* X[i])
				err = err + (label[i] - sum_)**2



			# atualizando os pesos através do erro acumulado
			self.pesos = self.pesos + (self.taxa_aprendizado * vetor_erro_acumulado)
			erro_agora = err/self.p
			epoch += 1
			print(f"Peso Epoch {epoch}: {self.pesos} Erro quadratico: {np.abs(erro_agora - erro_anterior)	}")
			self.histpeso[f"{epoch}"] = self.pesos
			self.histerr[epoch] = np.abs(erro_agora - erro_anterior)	



		return True
				

	def predict(self,X):
		'''Realizando predições com a rede neural'''

		X = np.hstack([-1*np.ones(X.shape[0]).reshape(-1,1),X])
		
		resultado = []
		for i in range(X.shape[0]):
			sum = (X[i]*self.pesos).sum()
			# Enviando para  a função de ativação
			y = self.FuncaoAtivacao(sum)
			resultado.append(y)
		
		return np.array(resultado)
			

	
