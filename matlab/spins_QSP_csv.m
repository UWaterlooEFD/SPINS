function [T1_max, T1_min, S1_max, S1_min, pdfCount] = spins_QSP_reader(filename)
  % Reads in the csv file and interprets it in the correct way for the user.
  %
  % Input: Filename (string) name of csv file.
  %
  % Output: Returns the max/min values for T1 and S1 (as specified in
  % spins.conf) and the (normalized) pdfCount histogram.
  A = importdata(filename);
  T1_max = A(1, 1);
  T1_min = A(1, 2);
  S1_max = A(1, 3);
  S1_min = A(1, 4);
  pdfCount = A(2:end, :);
  pdfCount /= sum(sum(pdfCount));
end
