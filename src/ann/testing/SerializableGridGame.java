package ann.testing;

import java.io.Serializable;
import java.util.List;

@SuppressWarnings("serial")
public class SerializableGridGame implements IGridGame, Serializable {
	private int rows;
	private int cols;
	private List<double[]> actionChoices;
	
	public SerializableGridGame() {}
	public SerializableGridGame(int rows, int cols, List<double[]> actionChoices) {
		this.rows = rows;
		this.cols = cols;
		this.actionChoices = actionChoices;
	}

	public static SerializableGridGame create(GridGame game) {
		return new SerializableGridGame(game.rows, game.cols, game.actionChoices);
	}
	
	public int getRows() {return rows;};
	public int getCols() {return cols;};
	public List<double[]> getActionChoices() {return actionChoices;}
}
