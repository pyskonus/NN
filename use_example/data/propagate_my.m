function propagate_my ()
  load tr_ex.dat;
  load theta1.dat;
  load theta2.dat;
  load theta3.dat;
  
  disp(size(theta1));
  disp(size(theta2));
  disp(size(theta3));
  disp(size(tr_ex));
  
  a = sigmoid(theta3*[ones(1, 1000); sigmoid(theta2*[ones(1,1000); sigmoid(theta1 * tr_ex)])]);
  
  a=a';
  save res.dat a;
  
endfunction
