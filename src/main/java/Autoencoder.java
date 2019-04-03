import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.ExponentialReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.*;

import static org.deeplearning4j.nn.conf.WorkspaceMode.*;

public class Autoencoder {

    private int _batchSizeTraining;
    private int _batchSizeTesting;
    private int _nEpochs;

    private int _numInputs;
    private int _numOutputs;

    String _filenameTrain;
    String _filenameTest;

    DataSetIterator _trainIter;
    DataSetIterator _testIter;

    private int[] _encoderSizes = new int[]{6,3};
    private int[] _decoderSizes = new int[]{3,6};

    private double _totalScore;
    private double _treshold;

    MultiLayerConfiguration _conf;

    MultiLayerNetwork _model;
    Evaluation _eval;
    org.deeplearning4j.nn.layers.variational.VariationalAutoencoder _vae;





    public Autoencoder() throws IOException, InterruptedException {

        _totalScore = 0;
        _batchSizeTraining = 1;
        _batchSizeTesting = 1;


        _nEpochs = 30;

        _numInputs = 10;
        _numOutputs = 10;


        _filenameTrain  = new ClassPathResource("OneClass/training_raul.csv").getFile().getPath();
        _filenameTest  = new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath();

        _trainIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/training_raul.csv").getFile().getPath(), _batchSizeTraining);
        _testIter = new AnomalyDataSetIterator(new ClassPathResource("OneClass/eval_raul.csv").getFile().getPath(), _batchSizeTesting);

        initModel();

        _eval = new Evaluation(_numOutputs);

    }

    private void initModel(){

        /*_conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(ENABLED).inferenceWorkspaceMode(ENABLED)
                .seed(123456)
                .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp.Builder().learningRate(0.05).rmsDecay(0.002).build())
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, input)
                .layer(1, encoder)
                //.layer(2, decoder)
                //.layer(3, output)
                .pretrain(true)
                .build();*/

        _conf = new NeuralNetConfiguration.Builder()
                .updater(new RmsProp.Builder().learningRate(0.05).rmsDecay(0.002).build())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new VariationalAutoencoder.Builder()
                        .activation(Activation.RELU)
                        .encoderLayerSizes(_encoderSizes)
                        .decoderLayerSizes(_decoderSizes)
                        .pzxActivationFunction(Activation.IDENTITY)
                        //.reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .reconstructionDistribution(new GaussianReconstructionDistribution(Activation.TANH))
                        .nIn(_numInputs)
                        .nOut(_numOutputs)
                        .build())
                .pretrain(true).backprop(false).build();


//        /System.out.println(_conf.toJson());



        _model = new MultiLayerNetwork(_conf);
        _model.init();


       _vae = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) _model.getLayer(0);


    }

    public void train() {
        for (int i = 0; i < _nEpochs; i++) {
            System.out.println("Epoch: "+i);
            _model.pretrain(_trainIter);    //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
        }

        //calculateThreshold();
    }

    private void calculateThreshold() {

        int trainDatacount=0;
        _trainIter.reset();
        while(_trainIter.hasNext()){
            trainDatacount++;
            DataSet t = _trainIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = _model.output(features,false);

            double score = calculateScore(predicted);
            System.out.println("Score: "+score);
            _totalScore += score;
        }

        _treshold = _totalScore/trainDatacount;
        System.out.println("Threshold: "+_treshold);
    }

    private double calculateScore(INDArray predicted) {
        return  Math.abs(predicted.data().getFloat(0)) +
                Math.abs(predicted.data().getFloat(1)) +
                Math.abs(predicted.data().getFloat(2));
    }

    public void evaluate() {

        List<Pair<Double,Integer>> evalList = new ArrayList<Pair<Double,Integer>>();
        int aux = 0;
        while(_testIter.hasNext()){
            DataSet t = _testIter.next();
            INDArray features = t.getFeatures();
            INDArray predicted = _model.output(features);
            predicted = _vae.generateRandomGivenZ(features, LayerWorkspaceMgr.noWorkspaces());
            //INDArray activate = _model.activate(features, true, LayerWorkspaceMgr.noWorkspaces());
            double score = calculateScore(predicted);

           // System.out.println("Label: "+labels+" Data: " + predicted+ " Score: "+score);



            evalList.add(new ImmutablePair<>(score, aux));
            aux++;
        }

        /*Collections.sort(evalList, Comparator.comparing(Pair::getRight));
        Stack<Integer> anomalyData = new Stack<>();

        System.out.println("Size: " + evalList.size());


        for (Pair<Double, Integer> pair: evalList) {
            double s = pair.getLeft();
            if (s >  _treshold) {
                anomalyData.push(pair.getRight());
            }
        }


        //output anomaly data
        System.out.println("based on the score, all anomaly data is following with descending order:\n");
        int anomalies = anomalyData.size();
        for (int i = anomalyData.size(); i > 0; i--) {
            System.out.println(anomalyData.pop());
        }

        System.out.println("Number of anomlies: " + anomalies);*/
    }

}