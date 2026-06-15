function unwrap_struct(struct)
% Opens structure "struct".
%
% Finds all of the names of the structure fields, 
% and then assigns all of those variables into the 
% workspace of the function that called unwrap_struct. 
%
% NOTE: This may cause there to be multiple MATLAB code-analyzer warnings of
% "Variable might be used before it is defined."
%
% This can be suppressed by calling unwrap_inputs with a warning-suppresser
% unwrap_struct(struct); %#ok<*NODEF>
%
%
% This can be also be useful for handling function vararg referenced as an
% options structure.
%
% For example, consider a function foo in foo.m, with one explicit
% argument, base_input, and and two optional arguments, X and Y.  The
% optional arguments can be given default values in an arguments section.
% Whether specified or not, X and Y can be processed as fields of a
% structure, options.
%
% function [out] = foo(base_input, options)
%     arguments 
%         base_input
%         options.X = 1
%         options.Y = 2
%     end
%     unwrap_inputs(options)
%     fprintf("%d, %d\n", X, Y)
% end
%
% >> foo(base_input) // Prints "1, 2\n"
% >> foo(base_input, Y=6) // Prints "1, 6\n".

names = fieldnames(struct);
for i = 1:numel(names);
    assignin('caller', names{i}, struct.(names{i}));
end

end % function unwrap_struct
