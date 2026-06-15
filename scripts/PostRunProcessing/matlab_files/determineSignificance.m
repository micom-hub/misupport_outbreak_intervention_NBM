function [ significant_prcc, significant_signifcance ] = determineSignificance(prcc, significance, alpha)

% Determine which PRCC values are significant

%
% TP is the number of timepoints.
%
% P is the number of varied parameters.
%
% O is the number of data columns (number of model outputs).

% Input:
%
%     prcc: A TP x O x P matrix of the PRCC values for the timesteps,
%           model outputs and varied model parameters.
%
%     significance: A TP x O x P matrix of the significance values for the
%                   PRCC values in prcc.
%
% alpha: The alpha value used to determine whether a PRCC value is significant.
%        If a PRCC significance value is < alpha the PRCC value is significant - 
%        that varied parameter is significant for that model output. 
%
% Output:
%
% significant_prcc: An TP x O x P matrix that is the same as prcc, except
%                   when a prcc value is not significant, according to the
%                   alpha test, that element of significant_prcc is 1.
%
% significant_signifcance: An TP x O x P matrix that is the same as
%                          significance, except when a significance value is
%                          not significant, according to the alpha test, that
%                          element of significant_signifcance is 1.

sizePrcc = size(prcc);
[TP, O, P] = size(prcc);
significant_prcc = ones(sizePrcc);
significant_signifcance = ones(sizePrcc);

for i = 1:1:TP
	for j = 1:1:O
	    for k = 1:1:P
	        if significance(i,j,k) < alpha
	            significant_prcc(i,j,k) = prcc(i,j,k);
	            significant_signifcance(i,j,k) = significance(i,j,k);
	        end
	    end
	end
end

end % function determineSignificance
