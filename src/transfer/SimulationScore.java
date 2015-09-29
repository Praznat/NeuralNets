package transfer;

import ann.Utils;
import modeler.ModelLearner;

public class SimulationScore {

	private String name;
	private double sum = 0;
	private final ModelLearner modeler;

	public SimulationScore(String name) {
		this.name = name;
		this.modeler = Utils.loadModelerFromFile(name, 0);
	}
	
	public SimulationScore(String name, ModelLearner modeler) {
		this.name = name;
		this.modeler = modeler;
	}

	public void observeNewScore(double score) {
		sum += score;
	}

	public ModelLearner getModeler() {
		return modeler;
	}
	
	public double getScore() {
		return sum;
	}
	
	@Override
	public String toString() {
		return name + "	" + sum;
	}

}
