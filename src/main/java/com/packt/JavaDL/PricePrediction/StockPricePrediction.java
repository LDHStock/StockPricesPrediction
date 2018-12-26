package com.packt.JavaDL.PricePrediction;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.NoSuchElementException;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.packt.JavaDL.PricePrediction.Representation.PriceCategory;
import com.packt.JavaDL.PricePrediction.Representation.StockDataSetIterator;
import com.packt.JavaDL.PricePrediction.Utils.PlotUtil;

import javafx.util.Pair;

import java.util.Scanner;

/**
 * StockPricePrediction Deep Learning
 * @author Daniel Fumero Cruz y Liam O'kelly Herrero
 * 
 */


public class StockPricePrediction {
	
	/**
	 * longitud de series de tiempo, suponga 22 días hábiles por mes
	 */
	
    private static int exampleLength = 22; //time series length, assume 22 working days per month
    private static StockDataSetIterator iterator;
    
	/**
	 * Main
	 * @param String[] args
	 * @return void
	 */ 	
	 
    public static void main (String[] args) throws IOException {
    	
    	/**
    	 * Se crea el lector
    	 */
    	
    	Scanner sc = new Scanner(System.in); //Se crea el lector
    	
    	/**
    	 * Se pide un dato al usuario
    	 */
    	
        System.out.print("Por favor ingrese el nombre del fichero `.csv`: "); //Se pide un dato al usuario
        
        /**
    	 * Se lee el nombre con nextLine() que retorna un String con el dato
    	 */
        
        String nombre = sc.nextLine(); //Se lee el nombre con nextLine() que retorna un String con el dato
        
        /**
    	 * cerramos el lector
    	 */ 	
        
        sc.close(); // cerramos el lector
        
        String file = "data/" + nombre;
        
        /**
    	 * nombre de stock
    	 */ 	
        
        String symbol = "GRMN"; // stock name
        
        /**
    	 * tamaño de mini lote
    	 */ 	
        
        int batchSize = 128; // mini-batch size
        
        /**
    	 * 80% para entrenamiento, 20% para pruebas.
    	 */ 	
        
        double splitRatio = 0.8; // 80% for training, 20% for testing
        
        /**
    	 * épocas de entrenamiento
    	 */ 	
        
        int epochs = 100; // training epochs

        System.out.println("Creating dataSet iterator...");
        
        /**
    	 * CERRAR: predecir precio cercano
    	 */ 	
        
        PriceCategory category = PriceCategory.OPEN; // CLOSE: predict close price
        iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        System.out.println("Loading test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        System.out.println("Building LSTM networks...");
        MultiLayerNetwork net = RecurrentNets.createAndBuildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        
        /**
    	 * Inicializar el backend de la interfaz de usuario.
    	 */ 	
        
       //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        
        /**
    	 * Configure dónde se almacenará la información de la red (gradientes, activaciones, puntaje vs. tiempo, etc.)
    	 * Luego agregue el StatsListener para recopilar esta información de la red, ya que se entrena
    	 * Alternativa: nuevo FileStatsStorage (Archivo) - vea UIStorageExample
    	 */ 	

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample

        /**
    	 * Adjunte la instancia de StatsStorage a la interfaz de usuario: esto permite que se visualicen los contenidos de StatsStorage
    	 */ 	
        
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        System.out.println("Training LSTM network...");
        for (int i = 0; i < epochs; i++) {
        	
        	 /**
        	 * modelo de ajuste utilizando datos de mini lotes
        	 */ 	
        	
            while (iterator.hasNext()) net.fit(iterator.next()); // fit model using mini-batch data
            
            /**
        	 * restablecer iterador
        	 */ 	
            
            iterator.reset(); // reset iterator
            
            /**
        	 * borrar estado anterior
        	 */ 	
            
            net.rnnClearPreviousState(); // clear previous state
        }
        
        /**
    	 * Imprima el número de parámetros en la red (y para cada capa)
    	 */ 	
        
        //Print the  number of parameters in the network (and for each layer)
		Layer[] layers_before_saving = net.getLayers();
		int totalNumParams_before_saving = 0;
		for( int i=0; i<layers_before_saving.length; i++ ){
			int nParams = layers_before_saving[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams_before_saving += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams_before_saving);

        System.out.println("Saving model...");
        File locationToSave = new File("data/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        
        /**
    	 * saveUpdater: es decir, el estado de Momentum, RMSProp, Adagrad, etc. Guarde esto para capacitar a su red más en el futuro
    	 */ 	
        
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        System.out.println("Restoring model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        
        /**
    	 * imprime la partitura con cada 1 iteración
    	 */ 	
        
        //print the score with every 1 iteration
        net.setListeners(new ScoreIterationListener(1));

        /**
    	 * Imprime el número de parámetros en la red (y para cada capa)
    	 */ 	
        
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

        System.out.println("Evaluating...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net, test, max, min);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net, test, max, min, category);
        }
        System.out.println("Done...");
    }    

    /**
	 * Predecir una característica de una acción de un día por delante
	 * @param Red Multi Capa "net", List<Pair<INDArray, INDArray>> testData, double max, double min, Categoría de precio category
	 * @return void
	 */ 	
    private static void predictPriceOneAhead (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];
        
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }
        
        RegressionEvaluation eval = net.evaluateRegression(iterator);   
        System.out.println(eval.stats());
        
        System.out.println("Printing predicted and actual values...");
        System.out.println("Predict, Actual");
        
        for (int i = 0; i < predicts.length; i++) 
        	System.out.println(predicts[i] + "," + actuals[i]);
        
        System.out.println("Plottig...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));
    }

    /**
	 * Predecir todas las características (abierto, cercano, bajo, precios altos y volumen) de una acción con un día de antelación
	 * @param Red Multi Capa "net", List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min
	 * @return void
	 */ 	
    private static void predictAllCategories (MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }
        
        System.out.println("Printing predicted and actual values...");
        System.out.println("Predict, Actual");
        for (int i = 0; i < predicts.length; i++) 
        	System.out.println(predicts[i] + "\t" + actuals[i]);
        System.out.println("Plottig...");
        
        RegressionEvaluation eval = net.evaluateRegression(iterator);   
        System.out.println(eval.stats());
        
        for (int n = 0; n < 5; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0: name = "Stock OPEN Price"; break;
                case 1: name = "Stock CLOSE Price"; break;
                case 2: name = "Stock LOW Price"; break;
                case 3: name = "Stock HIGH Price"; break;
                case 4: name = "Stock VOLUME Amount"; break;
                default: throw new NoSuchElementException();
            }
            PlotUtil.plot(pred, actu, name);
        }
    }
}
