package ann;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collection;

import modeler.ModelLearner;
import modeler.ModelLearnerHeavy;
import modeler.ModelerModule;
import modeler.TransitionMemory;


public class Utils {

	public static String stringArray(double[] ds, int e) {
		StringBuilder sb = new StringBuilder();
		for (double d : ds) {sb.append(round(d, e)); sb.append("	");}
		return sb.toString();
	}
	
	public static double oneOf(double[] ds) {
		return ds[(int) (Math.random() * ds.length)];
	}
	
	public static int round(double d) {
		return (int) Math.round(d);
	}
	public static double round(double d, int e) {
		final double t = (int) Math.pow(10, e);
		return Math.round(d * t) / t;
	}

	public static double between(double num, double denom, double lo, double hi) {
		return (num / denom) * (hi - lo) + lo;
	}
	
	public static double gaussianProbLE(double x, double variance) {
		return 0.5 * (1 + erf(x / Math.sqrt(2 * variance)));
	}
	
	public static double randomGaussianExpRate(double rate) {
		return Math.exp(rate * erf(Math.random()));
	}
	
	public static double erf(double z) {
		double t = 1 / (1 + 0.5 * Math.abs(z));
		double e = -z * z - 1.26551223 +
				t * (1.00002368 + 
				t * (0.37409196 +
				t * (0.09678418 +
				t * (-.18628806 +
				t * (0.2788607 +
				t * (-1.13520398 +
				t * (1.48851587 +
				t * (-.82215223 +
				t * (0.17087277)))))))));
		double ans = 1 - t * Math.exp(e);
		return z >= 0 ? ans : -ans;
	}
	
	public static double mean(double... x) {
		double sum = 0;
		for (double d : x) sum += d;
		return sum;
	}
	
	public static double stdev(double mean, double... x) {
		double sum = 0;
		for (double d : x) {
			double s = (d - mean);
			sum += s * s;
		}
		return Math.sqrt(sum / x.length);
	}
	
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(T[] array, T item) {
		return append((Class<T>)array.getClass().getComponentType(), array, item);
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(T[] array, T[] array2) {
		return append((Class<T>)array.getClass().getComponentType(), array, array2);
	}

	public static final <T> T[] appendItem(Class<T> clasz, T[] array, T item) {
		if (array == null || array.length == 0) {
			final T[] result = createArray(clasz, 1);
			result[0] = item;
			return result;
		} 
		
		final T[] result = createArray(clasz, array.length + 1);
		System.arraycopy(array, 0, result, 0, array.length);
		result[array.length] = item;
		return result;
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] append(Class<T> clasz, T[] array, T... items) {
		if(items != null && items.length == 1){
			return appendItem(clasz, array, items[0]);
		}
		
		if (array == null || array.length == 0) {
			return items;
		} 
		
		final T[] result = createArray(clasz, array.length + items.length);
		System.arraycopy(array, 0, result, 0, array.length);
		System.arraycopy(items, 0, result, array.length, items.length);
		return result;
	}
	@SuppressWarnings("unchecked")
	public static final <T> T[] createArray(Class<T> clasz, int len) {
		return (T[])Array.newInstance(clasz, len);
	}
	
	public static double[] concat(double[]... vs) {
		int len = 0;
		for (double[] v : vs) len += v.length;
		double[] result = new double[len];
		int i = 0;
		for (double[] v : vs) {
			System.arraycopy(v, 0, result, i, v.length);
			i += v.length;
		}
		return result;
	}
	
	public static double[] getActivations(Collection<? extends Node> nodes) {
		double[] result = new double[nodes.size()];
		int i = 0;
		for (Node n : nodes) result[i++] = n.getActivation();
		return result;
	}
	
    public static double gaussianPdf(double x) {
        return Math.exp(-x*x / 2) / Math.sqrt(2 * Math.PI);
    }

    public static double gaussianPdf(double x, double mu, double sigma) {
        return gaussianPdf((x - mu) / sigma) / sigma;
    }

    public static double wedgie(double x, double mu, double sigma, double base) {
        return Math.max(base, (1 - Math.abs(x - mu) / sigma));
    }
    
    public static double dWedgie(double x, double mu, double sigma, double base) {
    	final double a = Math.abs(x - mu);
    	return a / sigma + base >= 1 ? 0 : (mu - x) / (sigma * a);
    }

    /** elements of wheel must already add to 1 */
	public static int wheelOfFortuneDenomed(double[] wheel) {
		double b = Math.random();
		double cum = 0;
		for (int i = 0; i < wheel.length; i++) {
			cum += wheel[i];
			if (b < cum) return i;
		}
		throw new IllegalStateException("make sure wheel adds up to 1");
	}
    
	public static void saveModelerToFile(String namey, ModelLearner modeler) {
		if (namey.isEmpty()) return;
		saveNetworkToFile(namey+"P", modeler.getTransitionsModule().getNeuralNetwork());
		ModelerModule conditioner = modeler.getFamiliarityModule();
		if (conditioner != null) saveNetworkToFile(namey+"C", modeler.getFamiliarityModule().getNeuralNetwork());
	}
	
	/**
	 * name of stored modeler and new maximum experience size
	 */
	public static ModelLearnerHeavy loadModelerFromFile(String namey, int experienceSize) {
		if (namey == null || namey.isEmpty()) return null;
		FFNeuralNetwork storedP = Utils.loadNetworkFromFile(namey+"P");
		FFNeuralNetwork storedC = Utils.loadNetworkFromFile(namey+"C");
		if (storedP == null || storedC == null) {
			return null;
		} else {
			ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {},
					null, new int[] {}, ActivationFunction.SIGMOID0p5, experienceSize);
			modeler.getTransitionsModule().setANN(storedP);
			modeler.getFamiliarityModule().setANN(storedC);
			return modeler;
		}
	}
	
	public static boolean loadModelerFromFile(ModelLearner modeler, String namey) {
		if (namey.isEmpty()) return false;
		FFNeuralNetwork storedP = Utils.loadNetworkFromFile(namey+"P");
		FFNeuralNetwork storedC = Utils.loadNetworkFromFile(namey+"C");
		if (storedP == null || storedC == null) {
			return false;
		} else {
			modeler.getTransitionsModule().setANN(storedP);
			modeler.getFamiliarityModule().setANN(storedP);
			return true;
		}
	}
	
	public static void saveObjectToFile(String namey, Serializable data) {
		if (namey.isEmpty()) return;
		try {
			File file = new File("./saveFiles/"+namey);
			FileOutputStream fileOut = new FileOutputStream(file);
			if (!file.exists()) file.createNewFile();
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(data);
			out.close();
			fileOut.close();
			System.out.println("Serialized data is saved in " + namey);
		} catch(Exception e) {
			e.printStackTrace();
		}	
	}
	public static void saveNetworkToFile(String namey, FFNeuralNetwork ann) {
		saveObjectToFile(namey, ann);
	}
	
	public static Object loadObjectFromFile(String namey) {
		if (namey.isEmpty()) return null;
		try {
			FileInputStream fileIn = new FileInputStream("./saveFiles/"+namey);
			ObjectInputStream in = new ObjectInputStream(fileIn);
			Object result = in.readObject();
			in.close();
	        System.out.println("Loading data from " + namey);
			fileIn.close();
			return result;
		} catch(FileNotFoundException e) {
			System.out.println("No saved data found");
			return null;
		} catch(Exception e) {
			System.out.println(e);
			return null;
		}
	}
	@SuppressWarnings("unchecked")
	public static ArrayList<TransitionMemory> loadTrainingDataFromFile(String namey) {
		return (ArrayList<TransitionMemory>) loadObjectFromFile(namey);
	}
	public static FFNeuralNetwork loadNetworkFromFile(String namey) {
		return (FFNeuralNetwork) loadObjectFromFile(namey);
	}

	public static double sum(Collection<Double> ds) {
		double sum = 0;
		for (double d : ds) sum += d;
		return sum;
	}
	public static double correlation(ArrayList<Double> v1, ArrayList<Double> v2) {
		double sumDiff = 0;
		double ssq1 = 0;
		double ssq2 = 0;
		double avgV1 = sum(v1) / v1.size();
		double avgV2 = sum(v2) / v2.size();
		int n = Math.min(v1.size(), v2.size());
		for (int i = 0; i < n; i++) {
			double d1 = v1.get(i) - avgV1;
			double d2 = v2.get(i) - avgV2;
			sumDiff += d1 * d2;
			ssq1 += d1 * d1;
			ssq2 += d2 * d2;
		}
		return sumDiff / (Math.sqrt(ssq1) * Math.sqrt(ssq2));
	}

	public static boolean sameArray(double[] state, double[] otherState) {
		if (state.length != otherState.length) return false;
		for (int i = 0; i < state.length; i++) if (state[i] != otherState[i]) return false;
		return true;
	}

	public static double max(double[][] m) {
		double result = -666666;
		for (double[] v : m) {
			for (double d : v) result = Math.max(result, d);
		}
		return result;
	}
}
