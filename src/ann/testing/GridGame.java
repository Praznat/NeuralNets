package ann.testing;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;

import ann.ActivationFunction;
import ann.Utils;
import modeler.EnvTranslator;
import modeler.ModelLearnerHeavy;
import reasoner.Planner;
import reasoner.RewardFunction;

public abstract class GridGame {
	public ModelLearnerHeavy modeler;
	protected static JFrame frame = new JFrame();
	public static GridPanel gridPanel = new GridPanel(frame, 500);
	public static GridGameControlPanel controlPanel = new GridGameControlPanel(frame);
	static {
		frame.setLayout(new GridLayout(0,2));
		frame.add(gridPanel);
		frame.add(controlPanel);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
	}
	public final int rows;
	public final int cols;
	public final List<double[]> actionChoices = new ArrayList<double[]>();
	double[] chosenAction;
	RewardFunction rewardFn;
	boolean isPaused;
	final Thread thread;
	public GridGame(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		this.thread = Thread.currentThread();
	}
	protected static int getGridLoc(int col, int row, int[][] grid) {
		int cols = grid.length;
		int rows = grid[0].length;
		return grid[(col + cols) % cols][(row + rows) % rows];
	}
	public static double sum(double[][] grid) {
		double sum = 0;
		for (int c = 0; c < grid.length; c++) {
			for (int r = 0; r < grid[c].length; r++) {
				sum += grid[c][r];
			}
		}
		return sum;
	}
	public static double sum(double[] state) {
		double sum = 0;
		for (double d : state) sum += d;
		return sum;
	}
	public static double pctCorrect(int[][] grid1, double[][] grid2, boolean justOnes) {
		double sum = 0;
		int n = 0;
		for (int c = 0; c < grid1.length; c++) {
			for (int r = 0; r < grid1[c].length; r++) {
				int o1 = grid1[c][r];
				if (justOnes && o1 < 1) continue;
				double err = o1 - grid2[c][r];
				sum += Math.abs(err);
				n++;
			}
		}
		return n == 0 ? Double.NaN : Utils.round(1 - sum/n, 4) * 100;
	}
	
	public abstract double[] getState();
	
	public void convertFromState(double[] state) {
		System.out.println("Error: Need to implement (override) GridGame.convertFromState");
	}

	public abstract void oneTurn();
	
	public void setupForTurn() {}

	public RewardFunction getRewFun() {
		return rewardFn;
	}

	RewardFunction getRewardFunction() {
		return rewardFn;
	}
	void setRewardFunction(RewardFunction rewardFunction) {
		 this.rewardFn = rewardFunction;
	}
	
	public static void trainModeler(ModelLearnerHeavy modeler, int turns, GridGame game,
			int repaintMs, List<double[]> actionChoices, EnvTranslator actionTranslator) {
		Planner chimp = Planner.createRandomChimp();
		boolean repaint = repaintMs > 0;
		for (int t = 0; t < turns; t++) {
			long startMs = System.currentTimeMillis();
			double[] actionNN = chimp.getOptimalAction(game.getState(), actionChoices, 0, 0.1);
			game.chosenAction = actionTranslator.fromNN(actionNN);
			game.setupForTurn();
			modeler.observePreState(game.getState());
			modeler.observeAction(actionTranslator.toNN(game.chosenAction));
			game.oneTurn();
			frame.repaint();
			modeler.observePostState(game.getState());
			modeler.saveMemory();
			if (repaint) try {
				long elapsedMs = System.currentTimeMillis() - startMs;
				Thread.sleep(Math.max(0, repaintMs - elapsedMs));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	protected static ModelLearnerHeavy trainedModeler(int numVars, int numStates, GridGame game, double sampleSizeMultiplier,
			int repaintMs, List<double[]> actionChoices, EnvTranslator actionTranslator, int[] jdmHL) {
		int turns = (int) (numStates * sampleSizeMultiplier);
		System.out.println("Observing " + turns + " samples");
		ModelLearnerHeavy modeler = new ModelLearnerHeavy(500, new int[] {numVars*2},
				null, jdmHL, ActivationFunction.SIGMOID0p5, turns);
		trainModeler(modeler, turns, game, repaintMs, actionChoices, actionTranslator);
		return modeler;
	}
	
	protected abstract void paintGrid(Graphics g);
	
	public static void print(double[][] grid) {
		for (int r = 0; r < grid[0].length; r++) {
			String s = "";
			for (int c = 0; c < grid.length; c++) {
				s += Utils.round(grid[c][r], 2) + "	";
			}
			System.out.println(s);
		}
		System.out.println("***********");
	}
}

@SuppressWarnings("serial")
class GridPanel extends JPanel {
	final int gUnit;
	GridGame game;
	JFrame parent;
	public GridPanel(JFrame parent, int gUnit) {
		this.parent = parent;
		this.gUnit = gUnit;
	}
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		if (game != null) game.paintGrid(g);
	}
	public Dimension getPreferredSize() {
		if (game == null) return new Dimension(0, 0);
		return new Dimension(gUnit, gUnit);
	}
	public void setGame(GridGame game) {
		this.game = game;
		parent.pack();
	}
}
@SuppressWarnings("serial")
class GridGameControlPanel extends JPanel implements ActionListener {
	JFrame parent;
	GridGame game;
	JButton bL = new JButton("Predict-L");
	JButton bR = new JButton("Predict-R");
	JButton bU = new JButton("Predict-U");
	JButton bD = new JButton("Predict-D");
	JButton b1 = new JButton("Catch");
	JButton b2 = new JButton("Evade");
	public GridGameControlPanel(JFrame parent) {
		this.parent = parent;
		this.add(bL);
		this.add(bR);
		this.add(bU);
		this.add(bD);
		this.add(b1);
		this.add(b2);
		bL.addActionListener(this);
		bR.addActionListener(this);
		bU.addActionListener(this);
		bD.addActionListener(this);
		b1.addActionListener(this);
		b2.addActionListener(this);
	}
	public void setGame(GridGame game) {
		this.game = game;
		parent.pack();
	}
	@Override
	public void actionPerformed(ActionEvent e) {
		final String command = e.getActionCommand();
		if (command.startsWith("Predict")) {
			int joints = 3;
			double[] state = game.getState();
			double[] action = null;
			if (command.equals("Predict-L")) {
				action = GridExploreGame.LEFT;
			} else if (command.equals("Predict-R")) {
				action = GridExploreGame.RIGHT;
			} else if (command.equals("Predict-U")) {
				action = GridExploreGame.UP;
			} else if (command.equals("Predict-D")) {
				action = GridExploreGame.DOWN;
			}
			double[] newStateVars = game.modeler.newStateVars(state, action, joints);
//			System.out.println(Utils.stringArray(newStateVars, 2));
			game.convertFromState(newStateVars);
			parent.repaint();
//			game.convertFromState(state);
//			repaint();
		}
		else if (command.equals("Catch")) {
			game.setRewardFunction(GridTagGame.follow(game));
		}
		else if (command.equals("Evade")) {
			game.setRewardFunction(GridTagGame.evade(game));
		}
	}
}