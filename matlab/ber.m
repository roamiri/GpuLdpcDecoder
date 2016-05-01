% BER plot
snr=0.5:0.1:2.8;
error=[1.582e-1 1.550e-1 1.509e-1 1.462e-1 1.416e-1 1.328e-1 1.248e-1 1.134e-1 ...
    1.007e-1 8.762e-2 7.082e-2 5.555e-2 4.338e-2 2.965e-2 ...
    1.909e-2 1.224e-2 7.139e-3 4.230e-3 2.257e-3 1.024e-3 6.364e-4 ...
    2.736e-4 8.974e-5 3.281e-5];

berfit(snr,error);
ylabel('BER','FontSize',12,'FontWeight','bold','Color','k');
xlabel('Eb/N0(dB)','FontSize',12,'FontWeight','bold','Color','k');
% legend('1-Thread', '3-Threads','Location','northwest');
title('Bit Error Rate for AWGN Channel','fontsize', 12, 'FontWeight','bold');
grid on
set(gca, 'fontsize', 12, 'FontWeight','bold');
print('-djpeg','-r300', 'BER.jpg');