package deepnets.testing;

import java.awt.*;

import javax.swing.*;

import deepnets.Utils;

public abstract class GridGame {
	protected static JFrame frame = new JFrame();
	protected static GridPanel gridPanel = new GridPanel(frame, 500);
	protected static GridGameControlPanel controlPanel = new GridGameControlPanel(frame);
	static {
		frame.setLayout(new GridLayout(0,2));
		frame.add(gridPanel);
		frame.add(controlPanel);
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
	}
	protected final int rows, cols;
	public GridGame(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
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
		game.paintGrid(g);
	}
	public Dimension getPreferredSize() {
		if (game == null) return new Dimension(0, 0);
		return new Dimension(gUnit, gUnit);
	}
	void setGame(GridGame game) {
		this.game = game;
		parent.pack();
	}
}
@SuppressWarnings("serial")
class GridGameControlPanel extends JPanel {
	JFrame parent;
	GridTagGame game;
	JButton b1 = new JButton("Learn");
	JButton b2 = new JButton("Follow");
	JButton b3 = new JButton("Evade");
	public GridGameControlPanel(JFrame parent) {this.parent = parent;}
	void setGame(GridTagGame game) {
		this.game = game;
		parent.pack();
	}
}