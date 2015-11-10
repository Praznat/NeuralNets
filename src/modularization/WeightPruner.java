package modularization;

import java.util.*;

import ann.*;

public class WeightPruner {
	private static final Comparator<Connection> ABS_WGT_COMPARATOR = new Comparator<Connection>() {
		@Override
		public int compare(Connection c0, Connection c1) {
			return (int)Math.signum(Math.abs(c0.getWeight().getWeight()) - Math.abs(c1.getWeight().getWeight()));
		}
	};
	
	private final double pruneThresh;
	
	public WeightPruner(double pruneThresh) {
		this.pruneThresh = pruneThresh;
	}

	public void prune(FFNeuralNetwork ann) {
		prune(ann, pruneThresh);
	}
	
	public static void pruneBottomPercentile(FFNeuralNetwork ann, double pctile) {
		SortedSet<Connection> ordered = new TreeSet<Connection>(ABS_WGT_COMPARATOR);
		System.out.println("Pruning...");
		ordered.addAll(Connection.getAllConnections(ann));
		int n = (int) Math.round(ordered.size() * pctile);
		int i = 0;
		for (Connection conn : ordered) {
			if (i++ >= n) break;
			disconnect(conn);
		}
		System.out.println("Pruned " + (i-1) + " connections out of " + ordered.size());
	}

	public static void pruneBelowAvg(FFNeuralNetwork ann, double stdevsBelow) {
		Collection<Connection> conns = Connection.getAllConnections(ann);
		double avg = 0;
		for (Connection conn : conns) {
			double absW = Math.abs(conn.getWeight().getWeight());
			avg += absW;
		}
		avg /= conns.size();
		double var = 0;
		for (Connection conn : conns) {
			double diff = conn.getWeight().getWeight() - avg;
			var += diff * diff;
		}
		double stdev = Math.sqrt(var / conns.size());
		prune(ann, avg - stdevsBelow * stdev, conns);
	}
	public static void prune(FFNeuralNetwork ann, double pruneThresh) {
		prune(ann, pruneThresh, Connection.getAllConnections(ann));
	}
	public static void prune(FFNeuralNetwork ann, double pruneThresh, Collection<Connection> conns) {
		int n = 0;
		for (Connection conn : conns) {
			if (Math.abs(conn.getWeight().getWeight()) <= pruneThresh) {
				disconnect(conn);
				n++;
			}
		}
		System.out.println("Pruned " + n + " connections out of " + conns.size());
	}
	
	private static void disconnect(Connection conn) {
		conn.getInputNode().getOutputConnections().remove(conn);
		conn.getOutputNode().getInputConnections().remove(conn);
	}
	
	public static double[][] inOutAbsConnWgt(FFNeuralNetwork ann) {
		return inOutAbsConnWgt(ann, false);
	}
	
	/**
	 * returns a double[inputsSize][outputsSize] of absolute connection weights
	 * between inputs and outputs, printing to console optionally
	 */
	public static double[][] inOutAbsConnWgt(FFNeuralNetwork ann, boolean print) {
		ArrayList<? extends Node> inputs = ann.getInputNodes();
		ArrayList<? extends Node> outputs = ann.getOutputNodes();
		Map<Node, Integer> outToInt = new HashMap<Node, Integer>();
		for (int i = 0; i < outputs.size(); i++) outToInt.put(outputs.get(i), i);	
		double[][] result = new double[inputs.size()][outputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			storeCumAbsWgt(i, inputs.get(i), 0, outToInt, result);
		}
		if (print) for (int j = 0; j < result[0].length; j++) {
			for (int i = 0; i < result.length; i++) {
				System.out.print(Utils.round(result[i][j],4) + "	");
			}
			System.out.println();
		}
		return result;
	}
	
	private static void storeCumAbsWgt(int originI, Node n, double sofar,
			Map<Node, Integer> outToInt, double[][] result) {
		Collection<Connection> conns = n.getOutputConnections();
		if (conns.isEmpty()) {
			Integer o = outToInt.get(n);
			if (o != null) result[originI][outToInt.get(n)] += sofar;
		}
		else for (Connection conn : conns) {
			storeCumAbsWgt(originI, conn.getOutputNode(), sofar + Math.abs(conn.getWeight().getWeight()),
					outToInt, result);
		}
	}
}
