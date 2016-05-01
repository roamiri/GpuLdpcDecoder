% Scalablitiy
figure;
c_576_10 = [23 30; 23 33; 66 90; 94 127];
bar(c_576_10);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
% xlabel('Device','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(576,288), Iter=10','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_576_10.jpg');
% saveas(figure,'test.tif') % save image as a .tiff file
%%
figure;
c_2304_10=[23 31.5; 25 37; 66 94; 94 132];
bar(c_2304_10);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(2304,1152),Iter=10','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_2304_10.jpg');


%%
figure;
c_4k_10=[24 32; 27 34; 73 98; 98 131];
bar(c_4k_10);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(4000,2000),Iter=10','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_4k_10.jpg');


%%
% 5 Iterations
figure;
c_576_5 = [45 61; 44 61; 123 165; 163 217];
bar(c_576_5);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
% xlabel('Device','FontSize',12,'FontWeight','bold','Color','k');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(576,288), Iter=5','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_576_5.jpg');

%%
figure;
c_2304_5=[47 63; 47 63; 127 170; 170 226];
bar(c_2304_5);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(2304,1152), Iter=5','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_2304_5.jpg');


%%
figure;
c_4k_5=[44 60; 27 37; 139 169; 164 230];
bar(c_4k_5);
labs=['  K620  '; ' GT540M '; 'TeslaK20'; ' GTX680 '];
set(gca,'xticklabel',labs, 'fontsize', 12, 'FontWeight','bold');
ylabel('Throughput(Mbps)','FontSize',12,'FontWeight','bold','Color','k');
legend('1-Thread', '3-Threads','Location','northwest');
title('Throughput on Different Devices, code=(4000,2000), Iter=5','fontsize', 12, 'FontWeight','bold');
grid on
print('-djpeg','-r300', 'c_4k_5.jpg');







