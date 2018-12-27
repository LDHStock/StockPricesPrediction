package com.packt.JavaDL.PricePrediction;

import static org.junit.Assert.*;

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
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.packt.JavaDL.PricePrediction.Representation.PriceCategory;
import com.packt.JavaDL.PricePrediction.Representation.StockDataSetIterator;
import com.packt.JavaDL.PricePrediction.Utils.PlotUtil;

import javafx.util.Pair;

public class JunitTest {

	private static int exampleLength = 22; 
    private static StockDataSetIterator iterator;
    
	@Test
	public void testMain() throws IOException {
        String file = "data/prices-split-adjusted.csv";
        String symbol = "GRMN";
        int batchSize = 128; 
        double splitRatio = 0.8; 
        int epochs = 100; 

        System.out.println("Creating dataSet iterator...");
        PriceCategory category = PriceCategory.OPEN; 
        iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        System.out.println("Loading test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        System.out.println("Building LSTM networks...");
        MultiLayerNetwork net = RecurrentNets.createAndBuildLstmNetworks(iterator.inputColumns(), iterator.totalOutcomes());
        
   
        UIServer uiServer = UIServer.getInstance();


        StatsStorage statsStorage = new InMemoryStatsStorage();            
    
        uiServer.attach(statsStorage);
        
        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        System.out.println("Training LSTM network...");
        for (int i = 0; i < epochs; i++) {
            while (iterator.hasNext()) net.fit(iterator.next()); 
            iterator.reset(); 
            net.rnnClearPreviousState(); 
        }
       
      		Layer[] layers_before_saving = net.getLayers();
      		int totalNumParams_before_saving = 0;
      		for( int i=0; i<layers_before_saving.length; i++ ){
      			int nParams = layers_before_saving[i].numParams();
      			System.out.println("Number of parameters in layer " + i + ": " + nParams);
      			totalNumParams_before_saving += nParams;
      		}
      		System.out.println("Total number of network parameters: " + totalNumParams_before_saving);
      		
      	// test 
      		
      		assertEquals(68608,layers_before_saving[0].numParams());
      		
      		assertEquals(131584,layers_before_saving[1].numParams());
      		
      		assertEquals(131584,layers_before_saving[2].numParams());
      		
      		assertEquals(4128,layers_before_saving[3].numParams());
      		
      		assertEquals(1056,layers_before_saving[4].numParams());
      		
      		assertEquals(1056,layers_before_saving[5].numParams());
      		
      		assertEquals(33,layers_before_saving[6].numParams());
      		
      		int expected = 338049;
      		
      		assertEquals(expected,totalNumParams_before_saving);
      	  
      	// Fin test 

        System.out.println("Saving model...");
        File locationToSave = new File("data/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
    
        ModelSerializer.writeModel(net, locationToSave, true);

        System.out.println("Restoring model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
      
        net.setListeners(new ScoreIterationListener(1));

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
