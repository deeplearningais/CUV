#!/usr/bin/perl -w
use Data::Dumper;

my $g_idx = 0;

sub get_speed_test_binaries{
	my $path = shift();
	print "checking $path\n";
	my @l = glob("$path/*_speed");
    push @l, "$path/test_conv_op";
	return sort(@l);
}

sub get_test_results{
	my $dst = shift();
	my $test = shift();
	# rnd_uniform [fill_rnd_uniform(v)] took 3.2300 us/pass
    my $output = `$test`;
    for my $line (split(/\n/,$output)){
		my $is_dev = 0;
		$is_dev ||= $line =~ /\bdev_/;
		$is_dev ||= $line =~ /_dev\b/;
		next unless $is_dev;
		my ($took) = $line =~ /took\s+([\d.]+)/;
		my ($name) = $line =~ /\[(.*?)\(/;
		next unless $took and $name;
		$name = sprintf("%02d_$name", $g_idx++);
		$dst->{$name} = $took;
	}
}

sub run_all{
	my $path = shift();
	my @tests = get_speed_test_binaries($path);
	my %res;
	foreach (@tests){
		get_test_results(\%res, $_);
	}
	foreach my $k (sort(keys(%res))){
		print $k, "\t", $res{$k}, "\n";
	}
}

run_all($ARGV[0]);
