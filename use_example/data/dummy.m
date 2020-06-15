function dummy()
  load tr_ex.dat;
  %load theta1r.dat;
  %load theta2r.dat;
  %load theta3r.dat;
  load theta1.dat;
  load theta2.dat;
  load theta3.dat;
  load res.dat;
  frt = tr_ex(1,:);
  frr = res(1,:);
  
  a1 = frt;
  z2 = a1 * theta1';
  a2 = sigmoid(z2);
  z3 = [1, a2] * theta2';
  a3 = sigmoid(z3);
  z4 = [1, a3] * theta3';
  a4 = sigmoid(z4);
  
  d4 = a4 - frr;
  grad = d4' * [1, a3];
  %disp(d4);
  
  d3 = (theta3' * d4') .* sigmoidGradient([1;z3']);
  d2 = (theta2' * d3(2:end)) .* sigmoidGradient([1;z2']);
  
  disp(d4);
  disp(9);
  disp(d3);
  disp(9);
  disp(d2);
  
  
endfunction
