import * as tf from '@tensorflow/tfjs';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada da rede:
    // entrada de 7 posiçoes (idade normalizada, + 3 cores + 3 localizações)
    

    // 80 neurônios = aqui coloquei tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequententemen, mais processamento ele vai usar

    //A ReLu aage como um filtro:
    // E como se ele deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neuronio é positiva, passa para frent!
    //se for  zero ou negativa , pode jogar fora, na vai usar pra nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    //saida: 3 neuronios 
    // um para cada categoria (premiun, medium, basic)
    // activation: softmax normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    //compilando o modelo:
    // optmizer Adam (Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais:
    // ajustam os pesos de forma eficiente e inteligente
    // aprede com historico de erros e acertos

    //loss: categoricalCrossentropy = 
    // Ele compara o que o modelo "acha" (os scores de cada catagoria)
    // com a resposta certa
    //a categoria premin será sempre [1, 0, 0]

    //metrics: ['accuracy'] = quanto mais distante da previsão do modelo da resposata correta
    //maior e o erro (loss)
    //Exemplo Classico: classificação de imagens, recomendação, categorização de usuario
    //qualquer coisa em que a resposata certa e apenas uma entre varias possiveis.
    model.compile({ 
        optimizer: 'adam', 
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    //Treinamento do modelo
    //verbose: desabilita o log interno (e usa só callback)
    //epochs: quantidade de veses que vai rodar no dataset
    // shuffle: embaralha os dados, para evitar que o mode tenha vieses
    await model.fit(
        inputXs, 
        outputYs, 
        {
            verbose: 0,
            epochs:100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => {
                    console.log(`Epoch: ${epoch}: loss = ${log.loss}`);
                }
            }
        }
    );
    return model;
}
async function predict(model, pessoa){
    // transformar o array js para o tensor 9tfjs
    const tfInput = tf.tensor2d(pessoa)

    // faz a predição ( output será um vetor de 3 probabilidades)
    const pred =  model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({prob, index}));
}


    // Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//quanto mais dado melhor
//assim o algoritimo consegue entender melhor os padrões complexos
//dos dados de treino
const models = await trainModel(inputXs, outputYs);

const pessoa = {nome: "Erick", idade: 28, cor: "verde", localizacao: "Curitiba"};
// normalizando a idade da nova pessoal usando o mesmo padrão do treino
// Exemplo: idade_min= 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
const pessoaTensorNormalziada = [
    [
        0.2,
        1, // cor azul
        0, // cor vermelho
        0, // cor verde
        0, // localização São Paulo
        1, // localização Rio
        0 // localização Curitiba
    ]
]

const predictions = await predict(models, pessoaTensorNormalziada);
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n');

console.log(results)