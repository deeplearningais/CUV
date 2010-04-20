#!env perl

sub addlic_file{
	my $file = shift;
	my $tp   = shift;
	if (!&$tp($file)){
		return;
	}
	my ($owner, $orga, $year) = @_;
	local $/;
	undef $/;
	open FH, "<$file" or die $!;
	my $f = <FH>;
	close FH;
	my $haslic = $f =~ s|^//\*LB\*.*^//\*LE\*.*?$||ms;
	unless ($haslic) {
		print "$file is not licensed!\n";
		#return;
	}else{
		print "Relicensing $file...\n";
	}
	my $lic = get_lic($owner,$orga,$year);
	my $lf = "//*LB*\n$lic//*LE*\n$f";
	open FH, ">$file" or die $!;
	print FH $lf;
	close FH;
	#print $lf;
	#<STDIN>
}

sub addlic_dir{
	my $dir = shift;
	my $tp  = shift;
	my ($owner, $orga, $year) = @_;
	my @l = split /\0/, qx{find $dir -regex '.*\.\\(h\\|hpp\\|cuh\\|cu\\|cpp\\)\$' -print0};
	foreach my $f (@l){
		addlic_file($f, $tp, $owner,$orga,$year);
	}
}

sub get_lic{
	my ($owner, $orga, $year) = @_;
	local $/;
	undef $/;
	open LIC, "<scripts/LICENSE_TEMPLATE.txt" or die $!;
	my $lic = <LIC>;
	$lic =~ s/<OWNER>/$owner/g;
	$lic =~ s/<ORGANIZATION>/$orga/g;
	$lic =~ s/<YEAR>/$year/g;
	return $lic;
}

sub run{
	my @param = (
		"University of Bonn, Institute for Computer Science VI",
		"University of Bonn",
		"2010"
	);
	my $tp = sub{ $_=shift; !/3rd_party/ and !/third_party/};
	addlic_dir("src", $tp, @param);

	my @param = (
		"Alexander Krizhevsky",
		"University of Toronto",
		"2009"
	);
	my $tp = sub{ 1 };
	addlic_dir("src/3rd_party", $tp, @param);
}

run();
