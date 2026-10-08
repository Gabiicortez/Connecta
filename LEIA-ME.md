# IA Libras V5 — reconhecimento de letras e movimentos

A V5 substitui a lógica de detecção do app V4. **Não é necessário excluir nem sobrescrever suas gravações**. O código lê exemplos estáticos `.npy` de 126 características e gravações dinâmicas antigas de 120 × 126 características.

## 1. Como instalar no Windows

1. Faça uma cópia da pasta original do seu aplicativo antes de qualquer mudança.
2. Extraia este pacote em uma pasta nova (por exemplo `C:\IA_Libras_V5`).
3. **Copie a pasta `dataset_libras` da V4 para dentro dessa nova pasta**, ao lado dos arquivos `ialibras_v5.py` e `libras_core.py`. Se quiser testar com os modelos já treinados, copie também: `libras_static.pkl`, `libras_static_labels.json`, `libras_gru.keras` e `libras_dynamic_labels.json`, caso existam.
4. No terminal, dentro da pasta V5, execute:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements_libras.txt
python ialibras_v5.py
```

Se já usa Python 3.12, pode substituir `py -3.11` por `py -3.12`. A primeira operação de câmera pode baixar `hand_landmarker.task` automaticamente (é preciso conexão nessa primeira vez).

Não copie o `__init__.py` enviado junto com os arquivos originais: ele é um arquivo interno do **TensorFlow**, e **não faz parte do aplicativo**. Sua presença como `__init__.py` na pasta de testes pode interferir na execução do `pytest`.

## 2. Menu

1. Gravar letra/número estático.
2. Gravar gesto/palavra dinâmica (inclusive `NEUTRO`, para exemplificar movimento aleatório).
3. Listar gravações.
4. Treinar os novos modelos (V5).
5. Reconhecer em tempo real.
6. Excluir classe.
7. Sair.

Na janela de reconhecimento:

| Tecla | Função |
|---|---|
| `A` | Automático: letras paradas e gestos completos |
| `E` | Apenas letras/números, ideal para avaliar o alfabeto |
| `D` | Apenas gestos/palavras, ideal para avaliar os movimentos |
| `+` | Mais sensibilidade ao movimento (detecta movimentos menores) |
| `-` | Menos sensibilidade (ajuda contra falsos movimentos) |
| `C` | Limpar reconhecimento e iniciar outra tentativa |
| `Q` | Fechar a webcam |

O aplicativo **não tenta reconhecer um gesto a cada frame**. Primeiro observa o movimento e aguarda que ele termine. Para letras, espera cerca de meio segundo de mão estável e combina mais de uma previsão. Isso reduz a oscilação, mas não elimina erros de classificação do modelo.

## 3. Como melhorar a precisão com seus dados

- Grave **pelo menos 3 a 5 sessões diferentes por letra**; cada sessão coleta até 40 amostras. Mude luz, distância, posição e, se possível, a pessoa. Uma única sessão de 40 frames quase idênticos geralmente não é suficiente.
- Para gestos, grave **10 a 20 clipes completos por classe**, com a mão bem visível e diferentes velocidades. Evite deixar 2 segundos de repouso antes/depois da execução.
- Crie a classe dinâmica **`NEUTRO`** com exemplos de movimentos da mão *que não são sinais*, como ajustar posição ou chegar perto da câmera. Isso ajuda o modelo a não classificar todo movimento como uma palavra.
- Letras feitas por movimento — por exemplo, `J` e `Z` na datilologia — devem entrar no conjunto **dinâmico**, não no estático.
- É melhor treinar poucas classes bem registradas do que dezenas com uma única gravação.
- Treine novamente (opção 4) quando alterar as gravações. Os novos pesos V5 são salvos separadamente; a V4 é preservada.

Se a classe aparece errada mesmo no modo exclusivo `E` ou `D`, o problema provavelmente é de **treinamento/dados**, não de escolha entre modos. A pontuação do modelo não equivale necessariamente a uma probabilidade calibrada.

## 4. Compatibilidade

- Dados de entrada: `dataset_libras/estaticos/NOME/*.npy` e `dataset_libras/dinamicos/NOME/*.npy`.
- Modelos **V4** são utilizados quando ainda não existem modelos V5, com o formato de features original da V4.
- Modelos **V5** normalizam as características estáticas, pré-processam os vídeos para 48 frames e usam GRU menor. Eles não sobrescrevem os modelos V4.
- Se você iniciar a V5 **sem copiar o dataset**, ela criará pastas vazias. Não significa que seus exemplos antigos foram excluídos.
- Cópia de modelos V4 não garante que sejam compatíveis com todas as versões instaladas do Keras/TensorFlow. Se o carregamento falhar, copie os datasets e **treine com a V5**.

## 5. Desempenho e limitações

- A câmera lê frames continuamente e a IA pega o mais novo. O MediaPipe é executado com limite aproximado de ~18 análises por segundo, e o desenho da tela não exige detectar uma mão em cada atualização.
- A V5 calcula as posições da mão localmente para resistir melhor a deslocamento/zoom. O movimento preserva deslocamento temporal do gesto.
- O teste completo da webcam e do TensorFlow **não pôde ser executado** no ambiente de criação do pacote, que não tem acesso à webcam nem à instalação dessas bibliotecas. Existem testes automatizados de lógica e treinamento estático.
- A V5 reconhece **somente as classes treinadas**, não a Libras inteira. Sinais que dependem de expressões faciais, postura corporal, orientação e contexto não são totalmente modelados por 21 pontos das mãos.
- No Windows, versões modernas do TensorFlow normalmente usam CPU no sistema nativo. Treinamento CUDA usando sua GPU NVIDIA pode ser configurado via Linux/WSL2, mas **não é obrigatório** para testar a V5. Veja as orientações oficiais do TensorFlow.

## 6. Testar a lógica sem webcam

```powershell
python -m pytest -q testes_libras_v5.py
```

Os testes cobrem movimento/rejeição de letra durante movimento, estabilidade, ausência de mão, normalização, compatibilidade de gravação V4, separação de sessões para validação e treinamento estático com dados sintéticos.

## 7. Ajustes avançados

- Captura do MediaPipe: `PROCESS_INTERVAL` no `ialibras_v5.py` (padrão 0,055 segundos).
- Dimensões da detecção: `PROCESS_W`/`PROCESS_H` (384×288). Aumentar melhora visibilidade de mãos pequenas, mas exige mais CPU.
- Sensibilidade ao movimento: `start_threshold` e `stop_threshold` no `GestureRouter` em `libras_core.py` (ou use `+` e `-` durante o reconhecimento).
- Estabilidade de letras: `hold_seconds` e `StaticConsensus` em `libras_core.py`.
- Confiança mínima para palavra: 0,86 e margem 0,20 na função `realtime`. Ajuste só depois de avaliar **falsos positivos e falsos negativos** com vídeos reais.

**Observação:** a detecção de movimento no modo automático é heurística. Para uso real de tradução em Libras, será necessário obter um dataset grande, diverso, rotulado por pessoas que conheçam Libras, além de incluir informação corporal e facial quando pertinente.
