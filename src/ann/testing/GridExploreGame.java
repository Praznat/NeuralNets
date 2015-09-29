package ann.testing;

import java.awt.*;
import java.util.*;
import java.util.List;

import ann.*;
import reasoner.Planner;
import modeler.*;

public class GridExploreGame extends GridGame {

	private static final boolean WRAP = false;
	public static EnvTranslator actionTranslator = new EnvTranslator() {
		@Override
		public double[] toNN(double... n) {
			if (n[0] == -1.0) return new double[] {1,0,0,0};
			if (n[0] == 1.0) return new double[] {0,1,0,0};
			if (n[1] == -1.0) return new double[] {0,0,1,0};
			if (n[1] == 1.0) return new double[] {0,0,0,1};
			else return new double[] {0,0,0,0};
		}
		@Override
		public double[] fromNN(double[] d) {
			if (d[0] == 1) return new double[] {-1.0, 0}; // left
			if (d[1] == 1) return new double[] {1.0, 0}; // right
			if (d[2] == 1) return new double[] {0, -1.0}; // up
			if (d[3] == 1) return new double[] {0, 1.0}; // down
			else return new double[] {0.0};
		}
	};
	{
		actionChoices.clear();
		actionChoices.add(new double[] {1,0,0,0}); // LEFT
		actionChoices.add(new double[] {0,1,0,0}); // RIGHT
		actionChoices.add(new double[] {0,0,1,0}); // UP
		actionChoices.add(new double[] {0,0,0,1}); // DOWN
	}

	final int[][] playerGrid; // 1 represents location of player, 0 otherwise
	final int[][] walls; // these should not be input into modeler as they are static and derivable from transitions
	final double[][] smell;

	Point playerPos = new Point();
	protected int gUnit;

	public GridExploreGame(int rows, int cols) {
		super(rows, cols);
		playerGrid = new int[cols][rows];
		walls = new int[cols][rows];
		smell = new double[cols][rows];
		setPos(0,0,playerPos,playerGrid,false,false);
		gUnit = (int) Math.sqrt(cols*rows) * 20;
	}

	public static void main(String[] args) {
		int numtests = 100;
		int turns = 200;
		int size = 5;
		Collection<GEGTest> randos = new ArrayList<GEGTest>();
		Collection<GEGTest> modelers = new ArrayList<GEGTest>();
		for (int i = 0; i < numtests; i++) {
			GEGTest rando = new GEGTest();
			GEGTest modeler = new GEGTest();
			rando.test(size, turns, 1, true);
			modeler.test(size, turns, 0, true);
			randos.add(rando);
			modelers.add(modeler);
		}
		for (GEGTest t : randos) t.printEndResult();
		for (GEGTest t : modelers) t.printEndResult();
	}

	static class GEGTest {
		private ArrayList<Double> pctCompleteOverTime = new ArrayList<Double>();
		private ModelLearnerHeavy modeler;
		public ModelLearnerHeavy test(int size, int turns, double pctExploreFirst, boolean repaint) {
			BiasNode.clearConnections();
			int numPlanSteps = size-1;
			int numPlanRuns = 4;
			int memorySize = turns;
			int reflectTurns = 5;
			int learnIterations = 20;
			modeler = new ModelLearnerHeavy(500, new int[] {size * size * 2},
					new int[] {}, new int[] {}, ActivationFunction.SIGMOID0p5, memorySize);
			Planner chimp = Planner.createRandomChimp();
//			Planner explorer = Planner.createNoveltyExplorer(modeler, numPlanSteps, numPlanRuns, null, actionTranslator);
			Planner explorer = Planner.createKWIKExplorer(modeler, numPlanSteps, numPlanRuns,
					EnvTranslator.SAME, actionTranslator);
			GridExploreGame game = new GridExploreGame(size, size);
			gridPanel.setGame(game);
			double pctFilled = 0;
			for (int t = 0; t < turns; t++) {
				long startMs = System.currentTimeMillis();
				boolean rando = t < turns*pctExploreFirst;
				double[] actionNN = (rando ? chimp : explorer).getOptimalAction(game.getState(), game.actionChoices, 0, 0.1);
				modeler.observePreState(game.getState());
				modeler.observeAction(actionNN);
				game.move(actionTranslator.fromNN(actionNN), repaint);
				modeler.observePostState(game.getState());
				modeler.learnOnline(1.5, 0.5, 0); // instead of saveMemory
				if (!rando && (t+1) % reflectTurns == 0) {
					modeler.learnFromMemory(1.5, 0.5, 0, false, learnIterations);
				}
				// TODO store in array for graph over time
				pctFilled = game.pctFilled();
				pctCompleteOverTime.add(pctFilled);
				if (pctFilled >= 1) {
					printResult(pctExploreFirst > 0.99 ? "RANDO" : "MODELER", t, pctFilled);
					return modeler;
				}
				if (repaint) try {
					long elapsedMs = System.currentTimeMillis() - startMs;
					Thread.sleep(Math.max(0, 100 - elapsedMs));
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			printResult(pctExploreFirst > 0.99 ? "RANDO" : "MODELER", turns, pctFilled);
			return modeler;
		}
		private void printResult(String name, int turn, double pctFilled) {
			System.out.println(name + ":	" + pctFilled + "	at turn	" + turn);
		}
		private void printEndResult() {
			System.out.println(pctCompleteOverTime.get(pctCompleteOverTime.size()-1)
					+ "	at turn	" + (pctCompleteOverTime.size()+1));
		}
	}

	protected void setPos(int col, int row, Point pos, int[][] grid, boolean wrap, boolean clone) {
		if (!clone) grid[pos.x][pos.y] = 0;
		if (wrap) pos.setLocation((col + cols) % cols, (row + rows) % rows);
		else pos.setLocation(Math.min(Math.max(col, 0), cols-1),
				Math.min(Math.max(row, 0), rows-1));
		grid[pos.x][pos.y] = 1;
		smell[pos.x][pos.y] += 0.02;
	}

	@Override
	public void oneTurn() {
		move(chosenAction, false);
	}
	
	protected void move(double[] action, boolean repaint) {
		setPos(playerPos.x + (int) action[0], playerPos.y + (int) action[1], playerPos,playerGrid,WRAP,false);
		if (repaint) frame.repaint();
	}

	public double[] getState() {
		double[] result = new double[rows * cols];
		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) result[c + r*cols] = playerGrid[c][r];
		}
		return result;
	}

	private double pctFilled() {
		double sum = 0;
		for (double[] col : smell) {
			for (double d : col) if (d > 0) sum++;
		}
		return sum / (cols*rows);
	}
	
	protected boolean showSmell() {
		return true;
	}

	@Override
	protected void paintGrid(Graphics g) {
		final int gSub = gUnit/4;
		int[][] pGrid = this.playerGrid;
		for (int c = 0; c < pGrid.length; c++) {
			int[] col = pGrid[c];
			for (int r = 0; r < col.length; r++) {
				if (showSmell()) {
					int color = Math.max(0, (int) (255 - smell[c][r] * 256));
					g.setColor(new Color(color,color,color));
					g.fillRect(c*gUnit, r*gUnit, gUnit, gUnit);
				}
				g.setColor(Color.BLACK);
				g.drawRect(c*gUnit, r*gUnit, gUnit, gUnit);
				boolean player = col[r] > 0;
				if (player) g.fillOval(c*gUnit+gSub, r*gUnit+gSub, gUnit - 2*gSub, gUnit - 2*gSub);
			}
		}
	}

}
