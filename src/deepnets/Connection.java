package deepnets;

import java.io.Serializable;
import java.util.*;

@SuppressWarnings("serial")
public class Connection implements Serializable {

	private final Node inputNode, outputNode;
	private AccruingWeight weight;
	
	public Connection(Node inputNode, Node outputNode, AccruingWeight weight) {
		this.inputNode = inputNode;
		this.outputNode = outputNode;
		this.weight = weight != null ? weight : new AccruingWeight();
	}

	public static <N extends Node> Connection quickCreate(N inputNode, N outputNode, AccruingWeight weight) {
		Connection conn = new Connection((Node) inputNode, (Node) outputNode, weight);
		inputNode.getRawOutputConnections().add(conn);
		outputNode.getRawInputConnections().add(conn);
		return conn;
	}
	public static <N extends Node> Connection getOrCreate(N inputNode, N outputNode) {
		return getOrCreate(inputNode, outputNode, null);
	}
	public static <N extends Node> Connection getOrCreate(N inputNode, N outputNode, AccruingWeight weight) {
		Connection conn = null;
		for (Connection outputConn : inputNode.getRawOutputConnections()) {
			if (outputConn.getOutputNode() == outputNode) {
				conn = outputConn;
				break;
			}
		}
		if (conn != null) { // just check input-output = output-input
			boolean isOK = false;
			for (Connection inputConn : outputNode.getRawInputConnections()) {
				if (inputConn.getInputNode() == inputNode && inputConn == conn) isOK = true;
			}
			if (!isOK) System.out.println("CRITICAL ERROR: input-output != output-input");
		}
		if (conn == null) conn = quickCreate(inputNode, outputNode, weight);
		if (inputNode == outputNode) System.out.println("CRITICAL ERROR: node connected to itself");
		return conn;
	}
	
	public AccruingWeight getWeight() {
		return weight;
	}

	public Node getInputNode() {
		return inputNode;
	}

	public Node getOutputNode() {
		return outputNode;
	}

	@Override
	public String toString() {
		return inputNode.shortId() + "-" + outputNode.shortId() + " C" + Utils.round(weight.getWeight(), 2);
	}
	
	public static Collection<Connection> getAllConnections(FFNeuralNetwork ann) {
		Collection<Connection> result = new HashSet<Connection>();
		for (Layer<? extends Node> layer : ann.getLayers()) {
			for (Node n : layer.getNodes()) result.addAll(n.getOutputConnections());
		}
		return result;
	}
	
}
