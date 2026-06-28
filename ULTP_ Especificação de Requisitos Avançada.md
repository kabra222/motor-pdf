

Especificação Técnica e Arquitetura do
Motor Universal de Segmentação e
Caracterização Retórica de Documentos
## Jurídicos
- Escopo, Objetivos e Fundamentação Científica do
## Sistema
A transição de pipelines heurísticos baseados em expressões regulares para arquiteturas de
Processamento de Linguagem Natural (PLN) fundamentadas em aprendizado profundo
evidenciou falhas estruturais críticas no tratamento de documentos forenses em larga escala.
O fatiamento linear e cego de textos jurídicos — como contratos de fusões e aquisições,
estatutos sociais, petições iniciais e acórdãos — frequentemente destrói o nexo causal das
teses, isola cláusulas condicionais de seus núcleos de direito e aglutina matérias processuais e
materiais no mesmo segmento vetorial.
## 1
O Motor Universal de Segmentação e Caracterização Retórica de Documentos Jurídicos
(Universal Legal Text Parser - ULTP) propõe a adoção integral do paradigma da Abstração
Estrutural Probabilística. Esta arquitetura abandona definitivamente o conceito de quebra de
assunto como um fenômeno binário, rígido e dependente de palavras-chave, passando a tratar
a transição temática como uma flutuação topológica no espaço de estados latentes do
modelo de linguagem. O objetivo primordial consiste em projetar a arquitetura técnica e o
modelo de dados modular de um microsserviço (Python/TypeScript) capaz de realizar a
segmentação linear e hierárquica por tópicos, além da caracterização automática de atributos,
aplicável a uma variação de 200 a 500 tipologias de documentos jurídicos, eliminando
qualquer dependência de hard-coding de strings superficiais.
## 1
A Anatomia Biológica do Sentido e a Matemática Vetorial
Para a operacionalização do sistema por agentes de código e algoritmos de aprendizado de
máquina, o assunto ou tópico não pode ser definido empiricamente, mas deve ser circunscrito
matematicamente. O átomo do fluxo textual, denominado Unidade Semântica Atômica (ASU), é
estruturado a partir da intersecção de duas perspectivas complementares que unem a
linguística computacional à geometria de tensores.
## 1
A primeira perspectiva é a Célula Sintática Discursiva, frequentemente referida na literatura
avançada como Elementary Discourse Unit (EDU). A EDU constitui a menor oração dependente
ou independente indivisível, que deve ser processada abaixo do nível do ponto final, garantindo
que suas cadeias de correferência sejam plenamente resolvidas e vinculadas às suas entidades
de origem. A manipulação desta célula é rigidamente orientada por árvores de constituintes,
garantindo que relações de subordinação (como exceções e cláusulas de salvaguarda) nunca

sejam fisicamente separadas do núcleo obrigacional correspondente.
## 1
A segunda perspectiva repousa sobre o Vetor Atômico de Interação Tardia. Em vez de
depender de embeddings achatados que comprimem sentenças inteiras em um único vetor
global (como ocorre em arquiteturas Bi-Encoder tradicionais), a unidade mínima é projetada
dimensionalmente mantendo os tensores de cada token individual intactos.
## 3
## Esta
representação geométrica multidimensional do token individual é enriquecida dinamicamente
com os pesos de autoatenção de seus vizinhos locais nas camadas ocultas do Transformer,
permitindo que a similaridade máxima seja calculada de forma granular e preservando a
integridade das nuances processuais.
## 1
## Perspectiva
## Analítica
## Elemento
## Fundamental
Mecanismo de
Ação no ULTP
## Resultado Esperado
Linguística (Sintaxe) Célula Sintática
Discursiva (EDU)
Parsing de Árvore
de Constituintes
(Shift-Reduce)
Prevenção da
fragmentação de
orações
subordinadas e
condicionais.
## Geométrica
(Semântica)
Vetor Atômico de
## Interação Tardia
Manutenção da
matriz dimensional
do token
Preservação de
atributos locais
contra o
achatamento vetorial.
## Pragmática
(Discurso)
Resolução de
## Cadeias Anafóricas
## Mapeamento
intersentencial de
pronomes
Eliminação de
"Órfãos de Contexto"
e manutenção de
alvos originais.
- Eixo A: Ingestão Multimodal e Mitigação de
## Anisotropia
O processo de ingestão de documentos em sistemas de LegalTech frequentemente esbarra na
severa degradação introduzida por motores de Reconhecimento Óptico de Caracteres (OCR).
Documentos com layout complexo, como tabelas financeiras em estatutos sociais, notas de
rodapé extensas, fluxogramas corporativos ou carimbos verticais de tribunais, são

corrompidos quando linearizados de forma cega para a formatação de texto simples.
## 5
## Além
disso, a linguagem jurídica apresenta um alto índice de jargões repetitivos e formalidades
estilísticas de praxe, o que gera o colapso de anisotropia nos espaços vetoriais, um problema
crônico que precisa ser resolvido nas camadas iniciais da rede neural.
Ingestão Multimodal OCR-Free com Arquitetura ColPali
A abordagem arquitetural de fronteira exige a substituição completa dos pipelines OCR
clássicos por modelos de recuperação visual de documentos nativos. A integração do modelo
ColPali (Contextualized Late Interaction over PaliGemma) permite que as páginas dos arquivos
PDF sejam ingeridas diretamente como imagens densas.
## 6
A eliminação do estágio de
transcrição textual intermediária blinda o sistema contra os artefatos de digitalização.
O processamento visual é conduzido dividindo o documento em patches visuais uniformes,
tipicamente em uma grade configurada para projetar a geometria do documento em
representações independentes.
## 4
Um codificador de visão (Vision Encoder), fundamentado na
arquitetura SigLIP, processa esses patches, convertendo as características espaciais,
tipográficas e textuais em representações densas altamente contextuais.
## 9
## Posteriormente,
uma rede de projeção linear mapeia essas representações visuais diretamente no espaço de
linguagem latente de um modelo base (como o Gemma-2B), gerando múltiplos vetores por
página.
## 3
A inovação central desta abordagem reside no fato de que a recuperação e o cálculo de
similaridade não ocorrem de forma agregada, mas através do mecanismo de Interação Tardia
(Late Interaction) inerente ao paradigma ColBERT. O escore de similaridade entre um bloco
analítico (ou consulta) e o documento processado é calculado pela função MaxSim, definida
formalmente como:

Esta formulação matemática garante que a transição temática identifique mudanças
extremamente sutis no escopo de tabelas alfanuméricas, gráficos ou assinaturas, permitindo
que a similaridade entre o vetor da consulta e os vetores de fragmentos documentais atinja
uma resolução máxima sem perda de informação posicional ou estrutural.
## 4
Otimização de Escopo Longo com Modelos Mamba-Byte
A tokenização tradicional de sub-palavras (como o Byte Pair Encoding - BPE) é inerentemente
prejudicial a textos jurídicos. Essa limitação torna-se evidente ao lidar com hifenizações
corrompidas de processos antigos, códigos alfanuméricos longos (como chaves de blockchain
e hashes de assinatura digital), além da vasta gama de siglas institucionais específicas de
tribunais.
## 1
Para mitigar essa fragmentação prematura da semântica, o processamento contínuo
da arquitetura adota o Mamba-Byte, um modelo de espaço de estados (SSM) de seleção que

opera puramente no nível do byte.
## 12
A arquitetura Mamba elimina o custo computacional quadrático  associado ao
mecanismo de atenção densa dos Transformers tradicionais.
## 14
Ao tratar a sequência textual e
estrutural como uma série de bytes perfeitamente linear e contínua, o sistema mantém uma
memória de trabalho oculta (Hidden State) constante e excepcionalmente robusta contra
ruídos, dispensando o vocabulário fixo que engessa o reconhecimento de termos inéditos.
## 15
## A
implementação de decodificação especulativa, onde modelos paralelos rascunham e verificam
a inferência em nível de byte, viabiliza o processamento eficiente de códigos civis e regimentos
com dezenas de milhares de páginas.
## 13
O ganho de escala temporal assegura que a
integridade posicional dos tópicos em contextos super-longos seja mantida intacta, uma
premissa vital para a Abstração Estrutural Probabilística.
Regularização Vetorial SimCSE e Mitigação de Anisotropia
A presença maciça de expressões de praxe (como "Diante do exposto", "Termos em que pede
deferimento") e citações jurisprudenciais repetidas força os embeddings de sentenças a se
aglomerarem em uma região muito restrita do espaço dimensional. Esse fenômeno, conhecido
como Colapso de Anisotropia, faz com que a similaridade cosseno entre qualquer par de
frases jurídicas convirja invariavelmente para valores falsamente altos, tipicamente superiores a
, mesmo quando o objeto fático do argumento é diametralmente oposto.
## 1
A aplicação do SimCSE (Simple Contrastive Learning of Sentence Embeddings) de forma
não-supervisionada é um requisito mandatório para a mitigação desse efeito de
achatamento.
## 18
O aprendizado contrastivo atua passando a mesma sequência jurídica pelo
codificador duas vezes simultâneas, aplicando máscaras de dropout aleatórias e distintas no
mecanismo de atenção de cada passagem. Esta técnica engenhosa gera um par positivo
com pequenas flutuações latentes, enquanto todas as demais sentenças presentes
no mesmo lote de processamento (batch) são tratadas matematicamente como pares
negativos .
## 20
A função de perda InfoNCE (Noise Contrastive Estimation) é então
empregada para otimizar as matrizes de peso:

Essa penalização regulariza o espaço latente de maneira profunda, "achatando" a distribuição
singular dos vetores e restaurando a uniformidade geométrica que havia sido colapsada pela
repetição forense. Como resultado direto dessa expansão de espaço, os platôs artificiais de
similaridade contínua são estilhaçados. As descontinuidades semânticas que configuram reais
quebras de assunto tornam-se, assim, vales matemáticos nítidos, agudos e facilmente

detectáveis pelos motores de fronteira subsequentes.
## 17
- Eixo B: O Motor de Fronteira e Resolução de "Órfãos
de Contexto"
Após a geração de tensores de alta resolução com anisotropia estritamente corrigida e
contexto visual preservado, a etapa crítica da arquitetura reside em calcular onde executar
matematicamente as linhas de corte sem isolar entidades sintáticas fundamentais. A
fragmentação prematura que ignora referências anafóricas gera os indesejados "Órfãos de
Contexto", destruindo o índice de acerto em sistemas de busca e recuperação estruturada.
## 1
Detecção Bayesiana de Ponto de Mudança (BCPD) via Ruptures
Os algoritmos de segmentação textual da década de 90, como o TextTiling tradicional, falham
consistentemente em cenários modernos porque pressupõem limites estáticos de similaridade
e operam sobre tamanhos de blocos vocabulares invariáveis.
## 1
A especificação da nova
arquitetura requer abordagens probabilísticas e dinâmicas aplicadas a séries temporais
não-estacionárias. Para alcançar essa fluidez, o motor emprega a biblioteca
PyTorch-compatível ruptures, um framework avançado focado na detecção de pontos de
mudança offline.
## 21
A abordagem algorítmica principal utiliza a técnica PELT (Pruned Exact Linear Time). O método
PELT minimiza a soma do custo do erro empírico do segmento com uma penalidade linear
estrita, que é diretamente proporcional ao número de quebras inseridas, evitando a
super-segmentação e operando com custo computacional otimizado pela poda de caminhos
inviáveis.
## 23
Em vez de depender de distâncias fixas arbitradas manualmente, a detecção de
fronteiras via ruptures avalia os pontos longitudinais em que a variância estatística ou a média
vetorial dos embeddings sofre uma inflexão estrutural significativa.
## 22
A engenharia altamente modular do pacote permite a definição imperativa de classes
customizadas que herdam da estrutura BaseCost.
## 26
A customização desse cálculo matemático
é vital para o sistema. O modelo deve sobrescrever os métodos internos .fit() e .error(),
permitindo que a função de custo contemple a distribuição temporal intrínseca das matrizes e
avalie o comportamento estatístico da bacia local.
## 26
Assim, o algoritmo ganha autonomia para
inserir fronteiras de assunto apenas quando a distribuição entrópica do texto varia de forma
incontestável.
Otimização Geométrica Contínua com Distância de Wasserstein
Para capturar com precisão e elegância a magnitude do real "custo de mudança de assunto"
entre duas janelas adjacentes no fluxo do documento, a tradicional similaridade cosseno deve
ser suplantada pela formulação matemática de Transporte Ótimo (Optimal Transport). A
biblioteca Python Optimal Transport (POT) fornece a infraestrutura necessária para a
implementação da distância de Wasserstein (), permitindo mensurar a topologia
diferencial das distribuições de probabilidade de massa semântica geradas pelas sentenças.
## 29
A Distância Earth Mover's Distance (EMD) em uma dimensão reflete precisamente o custo

mínimo exigido para converter a massa semântica agregada de uma janela deslizante  no
formato distribucional da janela deslizante .
## 31

Para distribuições discretas, como a matriz temporal de sentenças jurídicas, o método eficiente
ot.wasserstein_1d aproveita a ordenação natural dos dados, entregando uma métrica
diferenciável .
## 32
O pico estatístico isolado gerado pela flutuação da métrica de
Wasserstein sinaliza o ponto absoluto de rompimento do nexo causal processual.
Diferentemente da similaridade isolada e plana, a lógica de transporte ótimo lida de forma
substancialmente superior com a paráfrase, com sinônimos longitudinais e com as digressões
doutrinárias intercaladas do autor do texto.
## 31
Proteção Gramatical por Árvores de Constituintes e Teoria Retórica
A aplicação direta de modelos puramente matemáticos e estatísticos sobre as sentenças
inevitavelmente resulta em "Órfãos de Contexto": a separação abrupta de uma frase contendo
o pronome demonstrativo "Este tribunal" do parágrafo imediatamente anterior que define de
fato a qual tribunal a entidade se refere.
Para solucionar essa degradação drástica downstream, o pipeline sintático do ULTP deve atuar
como uma parede intransponível de veto (guardrail). A adoção da biblioteca stanza fornece o
processamento neural robusto por algoritmos de shift-reduce, operando a conversão
profunda do texto em Árvores de Constituintes (Constituency Parse Trees) de alta resolução.
## 2
A análise de constituintes fornece um objeto hierárquico aninhado do tipo ParseTree, onde
nodos raízes e nós folhas estão encadeados por regras gramaticais estritas.
## 2
A lógica
determinística em código Python intercepta os vetores matemáticos e proíbe sumariamente o
fatiamento se o índice numérico da coordenada de corte incidir dentro de um sintagma
nominal complexo não resolvido ou violar a restrição fundamental da fronteira direita
(Right-Frontier Constraint), estabelecida pela Teoria da Estrutura Retórica (RST).
## 1
Conjuntamente, módulos especializados de rastreamento de correferência analisam
profundamente a cadeia anafórica do documento. Enquanto os pronomes processuais (como
"Reclamante", "Apelado", "A supramencionada lei") apontarem resolutamente para trás e
cruzarem o ponto de corte proposto pelo algoritmo ruptures, o motor é forçado
matematicamente a invalidar a quebra e fundir os blocos temporalmente, preservando a
irrestrita legibilidade do átomo de sentido.
## 1
Módulo do Eixo B Ferramenta Base Contribuição Mitigação de Risco

## Arquitetural Operacional
Cálculo de Bacias
## Temporais
ruptures
## (BCPD/PELT)
Abandono de
limites estáticos em
favor de variação
local.
## Evita
super-segmentação
em textos altamente
padronizados.
## Transporte Ótimo
## Semântico
POT (Wasserstein
## 1D)
Mensuração do
custo de mudança
do vocabulário
assimétrico.
Elimina o engano
causado por
paráfrases e
sinônimos no
cosseno.
## Guardião Sintático
## Retórico
stanza
(Constituency)
Preservação da
integridade de nós
dependentes e
subordinados.
Impede o surgimento
de orações
fragmentadas
incompreensíveis.
Rastreador de
## Anáforas
Módulo de
## Correferência
## Vinculação
temporal de
sujeitos ocultos e
pronomes de
tratamento.
Elimina os "Órfãos de
Contexto", blindando
o recall em RAG.
- Eixo C: Caracterização Hierárquica e Raciocínio
## Agêntico
Uma segmentação linear simples e plana, ainda que formulada de maneira matematicamente
impecável, sub-representa agressivamente a complexidade topológica de estatutos sociais e
manuais doutrinários. Nesses documentos maduros, os capítulos, as seções normativas e as
subcláusulas explicativas formam arranjos estruturais aninhados e interdependentes.
## 1
## Ignorar
essa tridimensionalidade acarreta um achatamento hierárquico desastroso para sistemas de
governança corporativa.

Regressão Ordinal Neuronal com Spacecutter
Para caracterizar corretamente a profundidade e a gravidade de uma quebra temática, a
decisão final do modelo de Inteligência Artificial deve ser modelada de forma intrinsecamente
ordinal, evitando o achatamento promovido pelas camadas de classificação categórica
tradicional. A arquitetura emprega a biblioteca spacecutter, que se acopla nativamente às
cabeças de classificação em redes neurais PyTorch, permitindo a inferência contínua e lógica
da magnitude hierárquica do corte (sendo, por exemplo, o Nível 1 representativo de um
Capítulo macro; o Nível 2 indicativo de uma Cláusula intermediária; e o Nível 3 relativo a um
Parágrafo micro-explicativo).
## 36
A arquitetura incorpora o OrdinalLogisticModel sobreposto à projeção de saída contínua do
codificador neural. Através do treinamento balizado pela CumulativeLinkLoss, o modelo não
apenas prevê um limite de categorias, mas impõe matematicamente que as margens ordinais
(cutpoints) permaneçam em estrita ordem sequencial e ascendente.
## 38
Uma previsão de corte
em profundidade de nível 3 engloba implicitamente as características limitadoras menores. A
implementação de callbacks rigorosos no ciclo de treinamento (como o AscensionCallback)
garante a eliminação de cruzamentos topológicos anômalos.
## 36
Esse rigor gera uma taxonomia
perfeitamente indexável em bancos vetoriais, propiciando navegação estruturada
intra-documental sem predições incompatíveis.
Extração Determinística Guiada por Gramáticas Livres de Contexto
## (CFG)
Após a detecção algorítmica rigorosa das fronteiras fáticas de um argumento, a caracterização
metadados do bloco resultante exige o uso de modelos de linguagem de grande porte (LLMs)
atuando como extratores analíticos lógicos. Modelos generativos puros, no entanto, estão
sujeitos a alucinação estrutural, falhando repetidamente em produzir dicionários JSON estritos
e compatíveis com os esquemas tipados da aplicação corporativa.
## 39
O motor de pesquisa resolve essa vulnerabilidade recomendando a integração profunda do
pacote outlines. Essa biblioteca transforma matrizes estruturais formais (como as classes de
validação do Pydantic ou JSON Schemas nativos) em Gramáticas Livres de Contexto (CFG)
formuladas no padrão EBNF (Extended Backus-Naur Form).
## 41
A gramática EBNF é então
compilada instantaneamente em um Autômato Finito (Finite State Machine) que filtra, mascara
e poda, em nível de tensores probabilísticos, todos os logits inadequados muito antes do
processo de amostragem durante a inferência do modelo generativo.
## 39
Com a diretriz sintática injetada no sampler do LLM:
## 1.
A rede neural é fisicamente proibida de gerar saídas que não obedeçam à hierarquia
estrita exigida pelas Tabelas Processuais Unificadas (TPU) ou pelos identificadores
numéricos do conselho de justiça.
## 1
## 2.
A extração de dados sensíveis (nomes de entidades regulatórias, valores de causas,
identificadores de leis) passa a ser executada via métodos de interface direta como
outlines.generate.json(), resultando em uma caracterização de atributos absolutamente

infalível e isenta de re-processamentos (retries) dispendiosos.
## 39
Personas e Loops Agênticos com Reasoning Tokens
A caracterização analítica de fragmentos contratuais de alta criticidade dogmática demanda
um escrutínio que transcende a classificação trivial. Com a orquestração de swarms de
agentes (Sistemas Multiagentes Cooperativos), o texto segmentado é submetido a diferentes
instâncias cognitivas que colaboram iterativamente visando atingir o equilíbrio ótimo de
consenso.
## 1
As arquiteturas de linguagem da nova geração que incorporam Tokens de Pensamento
(Reasoning Tokens, exemplificadas pelos modelos o1 da OpenAI ou pelas séries DeepSeek-R1)
produzem cadeias de lógica latente densas e transparentes. O fluxo metodológico instrui a IA a
extrair justificativas discursivas intrínsecas e debater as premissas antes de cravar a classe
jurídica definitiva do parágrafo. Esse ambiente assegura que as regras dogmáticas aplicadas
pelos agentes sejam formalmente confrontadas utilizando estratégias de crítica ortogonal.
Adicionalmente, a extração de Triplas Relacionais Atômicas (AERCs) durante este raciocínio
alimenta nativamente bancos de dados orientados a grafos (GraphRAG), interconectando os
assuntos fatiados através das entidades processuais extraídas. Para garantir o alinhamento
desse raciocínio estendido às preferências analíticas de advogados seniores, técnicas de
Otimização de Preferência de Razão de Chances (como ORPO - Odds Ratio Preference
Optimization) são aplicadas à etapa de fine-tuning. Diferentemente do RHLF clássico, o ORPO
funde a penalidade da geração indesejada diretamente à perda de entropia, alinhando a
caracterização do tópico sem o ônus de computar um modelo de recompensa separado.
## 44
- Eixo D: Validação Axiomática, Confiança e
Segurança em MLOps
Em operações corporativas sensíveis, a segmentação automatizada de informações
confidenciais ou laudos periciais não pode e não deve operar sem salvaguardas certificáveis
rigorosamente. A incerteza probabilística inerente aos sistemas de aprendizado de máquina
necessita de um envelopamento estatístico formal, transparência retroativa e restrições
lógicas inquebráveis antes da indexação final.
## 1
Predição Conforme e Intervalos de Confiança (Conformal Prediction)
A predição pontual (Point Prediction) exata de onde um tópico se divide ou a classificação
puramente argmax de seu atributo de risco carrega uma taxa de instabilidade intrínseca
inadmissível na auditoria forense. Para converter a inferência estatística bruta do modelo em
uma medida atestada e auditável de segurança jurídica, introduz-se a metodologia de Predição
Conforme (Conformal Prediction) ancorada na biblioteca Python crepes.
## 46
Esta tecnologia empacota os regressores ou classificadores base subjacentes (por meio de
construtos como WrapRegressor ou WrapClassifier) e utiliza um conjunto de calibração
mantido de forma independente para estimar empiricamente as pontuações de
não-conformidade do dado textual não visto.
## 48
Ao executar a função de classe
.predict_int(X_test, confidence=0.95), o pacote gera conjuntos ou intervalos de predição.
## 48

Em termos operacionais de MLOps: se o motor de fronteira calcular a linha de corte e a
incerteza estatística superar o nível aceitável, a biblioteca emite um intervalo contínuo
compacto indicando matematicamente que a transição temática real no documento ocorreu
inequivocamente entre a sentença  e a sentença , acompanhado de uma probabilidade
incondicional de acerto de .
## 51
Esta formulação de garantia previne cortes
excessivamente precisos mas incorretos, redirecionando automaticamente os blocos
estatisticamente marginais para auditoria humana assistida.
Camada Neuro-Simbólica por Programação Lógica (ASP)
A fronteira final do Guardrail Validador reside no acoplamento das saídas probabilísticas das
redes neurais a interpretadores rígidos de Programação em Lógica de Conjuntos (Answer Set
Programming - ASP), notoriamente através do solucionador de referência clingo.
## 53
A biblioteca clingo permite a execução fluida de lógicas declarativas simbólicas na forma de
grounding e solving diretamente emulados no fluxo de controle em Python.
## 55
O objetivo
primário dessa implementação é solidificar a ponte neuro-simbólica: enquanto os LLMs e
Encoders extraem e caracterizam de modo estocástico as propriedades de um documento, os
outputs resultantes (fatos inferidos, características extraídas, valores numéricos) são injetados
como variáveis isoladas de uma base de conhecimento simbólica. O método
clingo.Control.solve() submete todos esses predicados fáticos às restrições regulatórias do
projeto mapeadas rigidamente em ASP.
## 57
Se um bloco contratual for segmentado e classificado pelo LLM como pertencente a uma
jurisdição de "Dados Públicos", mas a extração relacional identificar a presença de hashes
bancários que indicam sigilo de dados sensíveis na lógica de conjuntos, o solucionador clingo
retorna instantaneamente um modelo nulo ou falha, barrando deterministicamente a injeção
do objeto no banco vetorial corporativo.
## 59
Este passo transmuta a natureza fundamentalmente
caixa-preta do Transformer em um processo de garantia de software transparente, provável e
estrito.
## 53
Explicabilidade Mecanicista do Modelo (XAI Forense)
Para que o MLOps da plataforma jurídica consiga comprovar conformidade com regulações de
Inteligência Artificial e rastreabilidade cibernética, as decisões das camadas latentes precisam
ser submetidas a uma minuciosa autópsia retrospectiva. A biblioteca captum atua em
integração visceral com a base PyTorch da rede, implementando o módulo de
LayerIntegratedGradients.
## 61
O algoritmo de Gradientes Integrados (Integrated Gradients) aproxima as derivadas ao longo
do caminho geométrico interpolado, estabelecendo um mapa de saliência preciso da
atribuição das entradas. Ele determina matematicamente e ilustra visualmente quais tokens
exatos e quais frases do texto jurídico carregaram o peso causador crítico para que o modelo
de regressão ordinal detectasse a presença da fronteira e categorizasse o bloco normativo.
## 63

Esta retroalimentação explicativa assegura total depuração da validade metodológica dos

hiperparâmetros de penalização.
Avaliação de Erros Longitudinais (Métricas Segeval)
A avaliação da qualidade e acurácia de sistemas de segmentação linear com o emprego de
métricas padrão e estáticas de classificação da academia (como -score, Precision e Recall)
resulta em diagnósticos míopes e equivocados. Na temporalidade da leitura, penalizar
duramente o modelo generativo como um erro completo por posicionar a quebra do assunto a
apenas uma sentença de distância do gabarito humano distorce o panorama de otimização da
rede.
Dessa forma, a integração mandatóriamente abriga a biblioteca segeval, focando nos cálculos
de erro de distância topológica e difusão através dos algoritmos WindowDiff e penalidade
## .
## 65
A implementação core da instrução segeval.window_diff(hypothesis, reference) movimenta
uma janela deslizante contínua pela série, quantificando as omissões e falsos positivos de
forma proporcional à aproximação geográfica inter-frase.
## 67
Tal abordagem valida e
recompensa as redes que convergem suas funções de perda de forma próxima e elegante às
zonas de transição fática real.
## 65
- Especificação Operacional Pronta para Agentes
Codificadores (SRS Core)
As seções a seguir extraem e materializam os vetores de pesquisa teóricos em topologia
modular arquitetural e artefatos de código restrito. Esta formatação garante que o imenso
escopo do ULTP possa ser lido, interpretado e fisicamente construído passo a passo por IAs de
codificação como Cursor, Antigravity e Codex, sem intervenção humana no design da
abstração.
6.1. Topologia Modular de Pastas do Projeto
A organização imperativa de diretórios reflete a separação estrita de interesses (Ingestão
Multimodal, Proteção Sintática, Motor Geométrico de Corte e Validador Neuro-Simbólico),
permitindo orquestração ágil e testabilidade assíncrona.
ultp-core/
├── src/
## │ ├── Ingestion/
│ │ ├── init.py
│ │ ├── colpali_retriever.py # Processamento Visual MaxSim e Ingestão via PaliGemma
│ │ ├── mamba_byte_stream.py # Processamento Linear Token-Free SSM
│ │ └── simcse_regularization.py # Perda InfoNCE para expansão do cone de Anisotropia
│ ├── SondaSintatica/
│ │ ├── init.py
│ │ ├── stanza_constituents.py # Árvores Shift-Reduce e Veto de Nós Constituintes
│ │ └── coreference_tracker.py # Resolução de Órfãos Contextuais via SpanBERT anafórico

│ ├── MotorFronteira/
│ │ ├── init.py
│ │ ├── bcpd_ruptures.py # Custom BaseCost e solver PELT Bayesiano
│ │ ├── wasserstein_distance.py # Distância 1D de Transporte Ótimo via POT
│ │ └── spacecutter_ordinal.py # Regressão Hierárquica Logística (CumulativeLinkLoss)
│ ├── ExtratorAtributos/
│ │ ├── init.py
│ │ ├── outlines_cfg_engine.py # Máscara EBNF e FSM constraints em logits
│ │ ├── pydantic_schemas.py # Mapeamento rigoroso Hierárquico dos Documentos
│ │ └── agent_swarm_logic.py # Orquestração LLM, Reasoning Tokens e GraphRAG
│ └── GuardrailValidador/
│ ├── init.py
│ ├── conformal_crepes.py # WrapRegressor, WrapClassifier e Sets de Predição
│ ├── clingo_asp_rules.py # Answer Set Programming: Restrições de Conformidade
│ └── captum_explainability.py # Auditoria de Saliência LayerIntegratedGradients
├── tests/
│ ├── test_boundaries_segeval.py # Telemetria contínua usando WindowDiff e P_k
│ ├── test_schemas_outlines.py
│ └── test_asp_neurosymbolic.py
├── requirements.txt
└── README.md
6.2. Protótipos de Código e Módulos Matemáticos Customizados
Os diagramas de software detalhados exigem matrizes de alta fidelidade que expressem a
geometria em tensores limpos e orientados a objeto. As especificações seguintes ancoram as
APIs.
A. Esquemas Pydantic Estritos para Hierarquia e Extração
(ExtratorAtributos/pydantic_schemas.py)
A caracterização gerada pelo LLM precisa ser envelopada sem brechas paramétricas. A
restrição intrínseca strict=True associada ao uso de enumerações predefinidas proíbe a
fabricação de metadados ilusórios.



## Python
from
pydantic
import
BaseModel, Field
from
typing
import
## List, Optional
from
enum
import
## Enum


class ClasseJuridicaDocumental(str, Enum):

## PETICAO_INICIAL =
## "peticao_inicial"

## CONTESTACAO =
## "contestacao"

## RECURSO =
## "recurso"

## ACORDAO_COLEGIADO =
## "acordao"

## CONTRATO_EMPRESARIAL =
## "contrato_social"

## ESTATUTO_HOLDING =
## "estatuto_corporativo"


class NivelHierarquiaAssunto(int, Enum):

## MACRO_CAPITULO =
## 1

## MEDIA_CLAUSULA =
## 2

## MICRO_EXPLICACAO =
## 3


class EntidadeProcessual(BaseModel):

nome_canônico:
str
= Field(..., description=
"Nome legal da entidade extraída (Pessoa Jurídica
ou Física)."
## )
papel_litigante: Optional[
str
## ] = Field(
## None
, description=
"Classificação no pólo (Ex: Autor,
## Réu, Outorgado)."
## )

class FragmentoTopicoValido(BaseModel):

id_unico_chunk:
str
= Field(..., description=
"Identificador UUID v4 da sequência textual."
## )
classe_raiz: ClasseJuridicaDocumental = Field(..., description=
"Tipologia documental
taxônomica oficial."
## )
profundidade_ordinal: NivelHierarquiaAssunto = Field(..., description=
"Profundidade
detectada pelo Spacecutter."
## )
sintese_raciocinio:
str
= Field(..., max_length=
## 200
, description=
"Explicação sintética da
virada fática do bloco."
## )
base_legal_citada: List[
str
## ] = Field(default_factory=
list
, description=
"Lista de artigos,
incisos e leis vinculadas."
## )
entidades_presentes: List[EntidadeProcessual] = Field(default_factory=
list
## ,
description=
"Agentes operantes referenciados."
## )
texto_processado:
str
= Field(..., description=
"A prosa do assunto segmentado com a cadeia
anafórica reconstruída."
## )

model_config = {

## "strict"
## :
## True
## ,

## "extra"
## :
## "forbid"

## }

B. Módulo Customizado de Custo para Algoritmo de Fronteiras
(MotorFronteira/bcpd_ruptures.py)
Para viabilizar a detecção estatística dos vales no solver ruptures, estabelece-se um objeto de

Custo Base derivado (BaseCost). Ele conjuga a divisão assimétrica dos sinais temporais com a
aplicação da métrica de similaridade fundamentada na distância geométrica de Transporte
Ótimo (Wasserstein).



## Python
import
numpy
as
np
import
ot
from
ruptures.base
import
BaseCost

class WassersteinTopologicalCost(BaseCost):


## """
Função de Custo personalizada para o motor Ruptures (BCPD/PELT).
Aplica Transporte Ótimo em 1D para quantificar o distanciamento de
massa semântica inter-janelas e acentuar a localização do vale fático.
## """

model =
## "custom_wasserstein_1d"

min_size =
## 3



def __init__(self):

self.signal =
## None

self.n_samples =
## None



def fit(self, signal):


""" Ingestão assíncrona do tensor temporal de representação do documento. """

self.signal = signal
self.n_samples = signal.shape

return
self


def error(self, start, end):


## """
Mensura a variância e a estabilidade da coesão do bloco [start:end].
Uma coesão disfuncional eleva dramaticamente o custo, forçando o PELT
a inserir uma fronteira matemática para regularizar a série.
## """


if
end - start < self.min_size *
## 2
## :

return

## 0.0


sub_tensor = self.signal[start:end]
mid_idx =
len
## (sub_tensor) //
## 2




# Desacoplamento da geometria de contexto para formulação das massas P e Q

magnitude_P = np.linalg.norm(sub_tensor[:mid_idx], axis=
## 1
## )
magnitude_Q = np.linalg.norm(sub_tensor[mid_idx:], axis=
## 1
## )


# Normalização de probabilidade distribucional local

P_dist = magnitude_P / np.
sum
(magnitude_P)
if
np.
sum
(magnitude_P)!=
## 0

else

magnitude_P
Q_dist = magnitude_Q / np.
sum
(magnitude_Q)
if
np.
sum
(magnitude_Q)!=
## 0

else

magnitude_Q


# Índices posicionais ordinais

coord_P = np.arange(
len
(P_dist), dtype=np.float64)
coord_Q = np.arange(
len
(Q_dist), dtype=np.float64)


# O solver de POT avalia o esforço de deformação entre as teses fáticas

w1_metric = ot.wasserstein_1d(coord_P, coord_Q, P_dist, Q_dist, p=
## 1
## )


# Retorno escalonado para viabilizar as podas lineares do framework Ruptures


return

float
(w1_metric * (end - start))

C. Esqueleto de Regressão Ordinal Logística PyTorch
(MotorFronteira/spacecutter_ordinal.py)
A representação hierárquica demanda que os cortes do documento não sejam processados
de maneira categórica avulsa. O uso do spacecutter impõe integridade temporal no cálculo
das perdas.



## Python
import
torch
import
torch.nn
as
nn
from
spacecutter.models
import
OrdinalLogisticModel
from
spacecutter.losses
import
CumulativeLinkLoss

class OrdinalHierarchicalSegmenter(nn.Module):


## """
Codificador profundo acoplado à arquitetura de Regressão Ordinal Logística.
Realiza o ranqueamento topológico assegurando aderência sequencial ascendente.

## """


def __init__(self, embedding_dimension: int, max_hierarchy_levels: int):


super
## ().__init__()

# Extrator contínuo de características latentes do bloco

self.latent_projector = nn.Sequential(
nn.Linear(embedding_dimension, embedding_dimension //
## 2
## ),
nn.GELU(),
nn.Dropout(p=
## 0.15
## ),
nn.Linear(embedding_dimension //
## 2
## ,
## 1
## )
# Convergência unidimensional para
ordinalidade

## )


# O Logistic Head cria os limites (cutpoints) escalonados entre os Níveis de Tópico

self.hierarchy_head = OrdinalLogisticModel(self.latent_projector,
max_hierarchy_levels)
self.base_criterion = CumulativeLinkLoss()


def forward(self, input_tensor):


return
self.hierarchy_head(input_tensor)


def evaluate_temporal_loss(self, predicted_logits, ground_truth, true_cut_idx, sample_idx):


## """
Incorporação de Penalização Temporal/Física: O custo da CumulativeLinkLoss
recebe penalização geométrica escalonada para que o backpropagation
aprenda a estabilizar nas franjas limítrofes exatas da transição do parágrafo.
## """

loss_val = self.base_criterion(predicted_logits, ground_truth)


# Fator de penalidade logarítmico em função da distância física do erro

decay_factor = torch.log1p(torch.
abs
(true_cut_idx - sample_idx).
float
## ())


return
loss_val * (
## 1.0
## +
## 0.1
- decay_factor)

# Atenção Sistêmica: Para treinamento efetivo, a chamada do optimizador deve obrigatoriamente

# ser sucedida por `model.apply(AscensionCallback())` para manter a integridade dos limites.


D. Gramática Formal Livre de Contexto EBNF
(ExtratorAtributos/ebnf_grammar.cfg)
A limitação da entropia de amostragem na camada de predição do LLM suprime a inserção
arbitrária de texto. A codificação BNF descrita força o roteamento das saídas diretamente para
o autômato finito do pacote outlines.




## EBNF
?start: json_strict_root

json_strict_root
: "{" whitespace "\"id_unico_chunk\"" whitespace ":" whitespace string_token
whitespace "," whitespace "\"classe_raiz\"" whitespace ":" whitespace schema_enum_classe
whitespace "," whitespace "\"profundidade_ordinal\"" whitespace ":" whitespace
num_limitado whitespace "," whitespace "\"sintese_raciocinio\"" whitespace ":" whitespace
string_token whitespace "}"

string_token
## : "\"" /[^"]*/ "\""
num_limitado
## : /[1-3]/
schema_enum_classe
## : "\"peticao_inicial\"" | "\"contestacao\"" | "\"recurso\"" | "\"acordao\"" |
## "\"contrato_social\"" | "\"estatuto_corporativo\""
whitespace
## : /[ \t\n]*/

6.3. Prompts de Sistema para Modulação de Cadeias Agênticas
A ativação do estado lógico do LLM exige matrizes instrucionais focadas em papéis cognitivos
distintos e alimentadas em tempo real com conhecimento vetorial ancorado em In-Context
## Learning.
Template de Injeção - Agente Validador de Raciocínio (Reasoning Critic):
Você atua como o Orquestrador Central de Validação Analítica Forense do projeto ULTP.
Sua especialidade primária é a caracterização semântica rigorosa e a validação topológica das
fronteiras temáticas sugeridas.
OBJETIVO IMUTÁVEL: Avalie criticamente o fluxo argumentativo do fragmento e valide as
premissas utilizando cadeias lógicas dedutivas antes de aprovar a extração do tópico e
fornecer o mapeamento JSON estruturado.
## PARÂMETROS DA CADEIA DE RACIOCÍNIO (CHAIN-OF-THOUGHT):
## 1.
Integridade de Constituintes: Examine a subordinação gramatical. Você jamais pode
homologar o fechamento de um segmento se ele isolar o núcleo principal do seu satélite
retórico (ex: quebrar antes de um "Desde que").
## 2.
Reconstrução Anafórica: Confirme se os elos coesivos de pronomes ou referências a
sujeitos ocultos não apontam exclusivamente para um contexto fora deste domínio
sugerido. Se ocorrer, a fronteira sugerida é considerada falsa e a classificação deve
agrupar o bloco adjacente.
## 3.
Classificação de Profundidade: Reflita sobre o peso jurídico para definir se o bloco
reflete uma Cláusula Matriz, um Parágrafo de Exceção ou um Resumo Doutrinário puro.
## CONTEXTO TEMPORAL EXTRAÍDO DO DOCUMENTO:

## {dynamic_context_window}
## CENTROIDES REFERENCIAIS DE ANCORAGEM (VETORES RECUPERADOS):
## {retrieved_examples_from_faiss_store}
Construa sua crítica na seção dedicada de pensamento interno e, tão logo atinja o equilíbrio de
Nash, imprima a resposta final rigidamente adstrita ao schema especificado, desprovida de
marcadores adicionais.
- Instrução Exata de Execução (Prompt de Ativação
do Deep Research)
Para deflagrar o processo integral de autoria assistida e orquestração autônoma da arquitetura
documentada, injete a seguinte instrução mestre no ambiente de trabalho das LLMs de
desenvolvimento. A declaração condensa os pilares metodológicos necessários:
"Execute uma investigação profunda de nível de Engenharia de Software Avançada, MLOps e
Direito Computacional com base nas diretrizes e nos requisitos de SRS descritos acima.
Vasculhe papers de conferências de ponta (como ICAIL, ACL, EMNLP, NLLP, SIGIR, ICLR) e
repositórios de produção de alta maturidade entre os anos de 2020 e 2026. Consolide os
achados em um documento técnico exaustivo e puramente operacional, fornecendo as
equações de perda, algoritmos geométricos, esquemas lógicos e a estrutura de pastas exata
detalhada na seção 3, de modo que assistentes de desenvolvimento em IA (Cursor, Antigravity,
Codex) consigam construir este motor universal de parsing passo a passo, de forma modular e
acoplada a guardrails determinísticos, sem a necessidade de intervenção humana para
decisões de design de arquitetura."
A aderência rigorosa a esta modelagem estabelece a ponte de transposição irrestrita do
conhecimento sintático de alto nível para a concretização determinística e programática da
extração da linguagem. O sistema atinge, consequentemente, a superação irrefutável dos
métodos heurísticos tradicionais aplicados aos complexos espaços latentes forenses.
Referências citadas
- gemini_conversation (2).pdf
- Constituency Parser - Stanza - Stanford NLP Group, acessado em maio 20, 2026,
https://stanfordnlp.github.io/stanza/constituency.html
- ColPali: Efficient Document Retrieval with Vision Language Models - arXiv,
acessado em maio 20, 2026, https://arxiv.org/html/2407.01449v4
- An Overview of Late Interaction Retrieval Models: ColBERT, ColPali, and ColQwen,
acessado em maio 20, 2026, https://weaviate.io/blog/late-interaction-overview
- ColPali: Efficient Document Retrieval with Vision Language Models - Hugging
Face, acessado em maio 20, 2026, https://huggingface.co/blog/manu/colpali
- The code used to train and run inference with the ColVision models, e.g. ColPali,
ColQwen2, and ColSmol. - GitHub, acessado em maio 20, 2026,
https://github.com/illuin-tech/colpali
- ColPali: Efficient Document Retrieval with Vision Language Models - arXiv,
acessado em maio 20, 2026, https://arxiv.org/html/2407.01449v1
- ColPali & Elasticsearch: How to search complex documents, acessado em maio

## 20, 2026,
https://www.elastic.co/search-labs/blog/elastiacsearch-colpali-document-search
- Late Interaction & Efficient Multi-modal Retrievers Need More Than a Vector
Index, acessado em maio 20, 2026,
https://www.lancedb.com/blog/late-interaction-efficient-multi-modal-retrievers-
need-more-than-just-a-vector-index
- ColPali: Efficient Document Retrieval with Vision Language Models - arXiv,
acessado em maio 20, 2026, https://arxiv.org/html/2407.01449v2
- Reproducibility, Replicability, and Insights into Visual Document Retrieval with Late
Interaction - arXiv, acessado em maio 20, 2026,
https://arxiv.org/html/2505.07730v1
- Paper page - MambaByte: Token-free Selective State Space Model - Hugging
Face, acessado em maio 20, 2026, https://huggingface.co/papers/2401.13660
- [2401.13660] MambaByte: Token-free Selective State Space Model - arXiv,
acessado em maio 20, 2026, https://arxiv.org/abs/2401.13660
- Transformer is Dead : Best Transformer alternates for LLMs | by Mehul Gupta |
Data Science in Your Pocket | Medium, acessado em maio 20, 2026,
https://medium.com/data-science-in-your-pocket/transformer-is-dead-best-tra
nsformer-alternates-for-llms-08e34f2846e3
- MambaByte: Token-free Selective State Space Model - OpenReview, acessado
em maio 20, 2026, https://openreview.net/forum?id=X1xNsuKssb
- From Bytes to Ideas: Language Modeling with Autoregressive U-Nets - arXiv,
acessado em maio 20, 2026, https://arxiv.org/html/2506.14761v1
- Addressing Asymmetry in Contrastive Learning: LLM-Driven Sentence
Embeddings with Ranking and Label Smoothing - MDPI, acessado em maio 20,
2026, https://www.mdpi.com/2073-8994/17/5/646
- Comparative Analysis of SimCSE for minBERT Optimization with Multiple
Downstream Tasks - Stanford University, acessado em maio 20, 2026,
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-r
eport-170040673.pdf
- [2104.08821] SimCSE: Simple Contrastive Learning of Sentence Embeddings -
arXiv, acessado em maio 20, 2026, https://arxiv.org/abs/2104.08821
- SimCSE: Simple Contrastive Learning of Sentence Embeddings - ACL Anthology,
acessado em maio 20, 2026, https://aclanthology.org/2021.emnlp-main.552.pdf
- Welcome to ruptures - GitHub Pages, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/
- Text segmentation - ruptures, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/examples/text-segmentation/
- Exact segmentation: Pelt — ruptures documentation - Index of, acessado em
maio 20, 2026,
https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/pelt.html
- Pelt - ruptures, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-refer
ence/
- Binary Segmentation: Entropy as a Cost Function - ServiceNow Security Lab,

acessado em maio 20, 2026,
https://securitylab.servicenow.com/research/2025-06-04-Binary-Segmentation-
Entropy-As-A-Cost-Function/
- Custom cost - ruptures, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costcustom/
- Fitting and prediction: estimator basics - ruptures, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/fit-and-predict/
- Custom cost function - ruptures, acessado em maio 20, 2026,
https://centre-borelli.github.io/ruptures-docs/custom-cost-function/
- Improving Cryo-EM Optimization Robustness with an Optimal Transport Loss
Function for Noisy Images | bioRxiv, acessado em maio 20, 2026,
https://www.biorxiv.org/content/10.64898/2025.12.23.696001v1.full-text
- Hands-on guide to Python Optimal Transport toolbox: Part 1 | Towards Data
Science, acessado em maio 20, 2026,
https://towardsdatascience.com/hands-on-guide-to-python-optimal-transport-t
oolbox-part-1-922a2e82e621/
- POT: Python Optimal Transport - Journal of Machine Learning Research,
acessado em maio 20, 2026,
https://jmlr.csail.mit.edu/papers/volume22/20-451/20-451.pdf
- Quick start guide - POT: Python Optimal Transport, acessado em maio 20, 2026,
https://pythonot.github.io/quickstart.html
- Source code for ot.lp.solver_1d - POT: Python Optimal Transport, acessado em
maio 20, 2026, https://pythonot.github.io/_modules/ot/lp/solver_1d.html
- A Semantic Parsing Algorithm to Solve Linear Ordering Problems - arXiv,
acessado em maio 20, 2026, https://arxiv.org/html/2502.08415v1
- Data Objects and Annotations - Stanza - Stanford NLP Group, acessado em maio
20, 2026, https://stanfordnlp.github.io/stanza/data_objects.html
- spacecutter - Ordinal regression models in PyTorch - GitHub, acessado em maio
20, 2026, https://github.com/EthanRosenthal/spacecutter
- spacecutter-torch - PyPI, acessado em maio 20, 2026,
https://pypi.org/project/spacecutter-torch/
- spacecutter: Ordinal Regression Models in PyTorch | Ethan Rosenthal, acessado
em maio 20, 2026,
https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/
- Outlines: structured JSON/regex/Pydantic LLM generation | Hermes Agent,
acessado em maio 20, 2026,
https://hermes-agent.nousresearch.com/docs/user-guide/skills/optional/mlops/ml
ops-inference-outlines
- dottxt-ai/outlines: Structured Outputs - GitHub, acessado em maio 20, 2026,
https://github.com/dottxt-ai/outlines
- Structured Outputs - vLLM, acessado em maio 20, 2026,
https://docs.vllm.ai/en/latest/features/structured_outputs/
- outlines 0.2.0 - PyPI, acessado em maio 20, 2026,
https://pypi.org/project/outlines/0.2.0/
- Generate structured output from LLMs with Dottxt Outlines in AWS | Artificial

Intelligence, acessado em maio 20, 2026,
https://aws.amazon.com/blogs/machine-learning/generate-structured-output-fro
m-llms-with-dottxt-outlines-in-aws/
- ORPO Trainer - Hugging Face, acessado em maio 20, 2026,
https://huggingface.co/docs/trl/v0.9.4/orpo_trainer
- ORPO Trainer - Hugging Face, acessado em maio 20, 2026,
https://huggingface.co/docs/trl/en/orpo_trainer
- crepes — crepes v. 0.9.0, acessado em maio 20, 2026,
https://crepes.readthedocs.io/
- The crepes package — crepes v. 0.9.0 - Read the Docs, acessado em maio 20,
2026, https://crepes.readthedocs.io/en/latest/crepes.html
- henrikbostrom/crepes: Python package for conformal prediction · GitHub,
acessado em maio 20, 2026, https://github.com/henrikbostrom/crepes
- Examples — crepes v. 0.9.0, acessado em maio 20, 2026,
https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html
- Embracing Uncertainty with Conformal Prediction | Blog post by Robbert van
Kortenhof, acessado em maio 20, 2026,
https://bigdatarepublic.nl/articles/embracing-uncertainty-with-conformal-predict
ion/
- crepes: a Python Package for Generating Conformal Regressors and Predictive
Systems - Proceedings of Machine Learning Research, acessado em maio 20,
2026, https://proceedings.mlr.press/v179/bostrom22a/bostrom22a.pdf
- Conformal Prediction in Python with crepes - GitHub, acessado em maio 20,
## 2026,
https://raw.githubusercontent.com/mlresearch/v230/main/assets/bostrom24a/bos
trom24a.pdf
- A Survey of Neurosymbolic Answer Set Programming, acessado em maio 20,
2026, https://neurosymbolic-ai-journal.com/system/files/nai-paper-877.pdf
- Answer Set Programming for Legal Analysis - Python for Law, acessado em maio
20, 2026, https://pythonforlaw.com/2025/08/25/answer-set-programming.html
- clingo.control API documentation - Potassco, acessado em maio 20, 2026,
https://potassco.org/clingo/python-api/5.8/clingo/control.html
- Python: built-in module clingo - Potassco, acessado em maio 20, 2026,
https://potassco.org/clingo/python-api/5.1/clingo.html
- clingo.solving API documentation - Potassco, acessado em maio 20, 2026,
https://potassco.org/clingo/python-api/5.5/clingo/solving.html
- Integrating ASP into ROS for Reasoning in Robots - Semantic Scholar, acessado
em maio 20, 2026,
https://pdfs.semanticscholar.org/265f/090d701a0489ecfbb2e7df3bc2ca18871f3a.
pdf
- A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for
Autonomous Driving Planning - arXiv, acessado em maio 20, 2026,
https://arxiv.org/html/2603.12421v1
- Specifying Goals to Deep Neural Networks with Answer Set Programming - AAAI
Publications, acessado em maio 20, 2026,

https://ojs.aaai.org/index.php/ICAPS/article/download/31454/33614/35511
- Interpreting text models: IMDB sentiment analysis - Captum, acessado em maio
20, 2026, https://captum.ai/tutorials/IMDB_TorchText_Interpret
- Visual Question Answering Model - Captum · Model Interpretability for PyTorch,
acessado em maio 20, 2026,
https://captum.ai/tutorials/Multimodal_VQA_Interpret
- Integrated Gradients - Captum · Model Interpretability for PyTorch, acessado em
maio 20, 2026, https://captum.ai/api/integrated_gradients.html
- LLM Attribution - Captum · Model Interpretability for PyTorch, acessado em maio
20, 2026, https://captum.ai/tutorials/Llama2_LLM_Attribution
## 65.
segeval · PyPI, acessado em maio 20, 2026, https://pypi.org/project/segeval/
- An Alignment-based Approach to Text Segmentation Similarity Scoring - ACL
Anthology, acessado em maio 20, 2026,
https://aclanthology.org/2022.conll-1.26.pdf
- emnlp2015-ih-ig/code/experiments/src/main/python/segeval/window/windowdiff.
py at master, acessado em maio 20, 2026,
https://github.com/UKPLab/emnlp2015-ih-ig/blob/master/code/experiments/src/m
ain/python/segeval/window/windowdiff.py
- Segmentation Evaluation Documentation, acessado em maio 20, 2026,
https://segeval.readthedocs.io/_/downloads/en/latest/pdf/
- Quickstart — SegEval v2.0.11 Documentation - Segmentation Evaluation using
SegEval, acessado em maio 20, 2026,
https://segeval.readthedocs.io/en/latest/user/quickstart/