#!/bin/env perl

use warnings;
use Template;

my $sfile = shift @ARGV;
my $ifile = shift @ARGV;
my $ofile = shift @ARGV;
my $kernel_name = shift @ARGV;

print "Generating $ofile using $ifile\n";

open(PFILE, "<$sfile") or die $!;

my ($max_num_imgs_at_once, @seq_row_fact, $spmm_block_size);
while(<PFILE>){
	/^#define/ and print;
	/^#define\s*MAX_NUM_IMGS_AT_ONCE\s*(\d+)\s*/ and $max_num_imgs_at_once = $1;
	/^#define\s*SEQ_ROW_FACT\s*(\d[,\d]*)\s*/    and @seq_row_fact = split(/,/,$1);
	/^#define\s*SPMM_BLOCK_SIZE\s*(\d+)\s*/      and $spmm_block_size = $1;
}
print<<EOT
- MAX_NUM_IMGS_AT_ONCE: $max_num_imgs_at_once
- SEQ_ROW_FACT:         @seq_row_fact
- SPMM_BLOCK_SIZE       $spmm_block_size

EOT
;


open IFILE, "<$ifile" or die $!; $/=undef;
my $templ = <IFILE>;
my $outstr = "";
my (@ifclauses,@ifclausesTrans);

my $tt = Template->new({
		INTERPOLATE  => 1,
	}) || die "$Template::ERROR\n";

foreach my $rf (@seq_row_fact){
foreach my $ni (1..$max_num_imgs_at_once){
	print "Instantiating rf=$rf ni=$ni\n";
	my $tmpl2 = $templ;

	# now expand the loops et cetera
	my $vars = {
		nimgs  => [(0..($ni-1))],
		ni     => $ni,
		bs     => $spmm_block_size,
		rf     => $rf,
		rfs    => [(0..($rf-1))],
	};
	my $tmpl3;
	$tt->process(\$tmpl2, $vars,\$tmpl3) || die $tt->error(), "\n";
	$tmpl2 = $tmpl3;

	my $o =<<"EOT";
#if defined NUM_IMG
#   undef NUM_IMG
#endif
#if defined BLOCK_SIZE
#   undef BLOCK_SIZE
#endif
#if defined ROW_FACT
#   undef ROW_FACT
#endif
#define NUM_IMG     $ni
#define BLOCK_SIZE  $spmm_block_size
#define ROW_FACT    $rf
$tmpl2

#undef BLOCK_SIZE
#undef ROW_FACT
#undef NUM_IMG
EOT
$outstr .= $o;

my $kernel = (($ni < 0) ? "spmm_${kernel_name}_kernel_trans_register" : "spmm_${kernel_name}_kernel_trans_shared");
$kernel .= "_" . join("_",($spmm_block_size, $ni, $rf));
my $has_stride = $kernel_name eq "dia";
my $stride_param = ( $has_stride )?"A.stride(),":"A.input_maps(),A.output_maps(),";
my $ifc = "\t\telse if(nimg == $ni){
               $kernel<value_type, index_type, true,true,true> <<<grid, $spmm_block_size>>> (A.h(), A.w(),  A.num_dia(),  $stride_param A.get_offsets().ptr(), A.vec().ptr(), v.ptr(), dst.ptr(), factAv,factC,toff);
		   }";
push @ifclausesTrans, $ifc;
$ifc =~ s/_trans//g;
push @ifclauses, $ifc;
}
}




$outstr .=<<"EOT";
template<class value_type, class index_type>
void spmm_device_${kernel_name}_dispatch(const ${kernel_name}_matrix<value_type,dev_memory_space,index_type>& A, 
					const vector<value_type,dev_memory_space>& v, 
					vector<value_type,dev_memory_space>& dst, 
					char transA,
					const value_type& factAv,
					const value_type& factC,
					const unsigned int& toff){
	if(transA=='n'){
		const dim3 grid = make_large_grid(A.h(),$spmm_block_size);
#if BLOCK_SIZE_LIMITS_NUM_DIAG
		cuvAssert(A.num_dia() <= $spmm_block_size); // kernel doesn't handle larger numbers of diagonals
#endif
	    int nimg = v.size() / A.w();
		if(0);
		@ifclauses
		else cuvAssert(false);
	}else if(transA=='t'){
	    int nimg = v.size() / A.h();
		const dim3 grid = make_large_grid(A.w(),$spmm_block_size);
#if BLOCK_SIZE_LIMITS_NUM_DIAG
		cuvAssert(A.num_dia() <= $spmm_block_size); // kernel doesn't handle larger numbers of diagonals
#endif
		if(0);
		@ifclausesTrans
		else cuvAssert(false);
	}else{
	    cuvAssert(false);
	}
	cudaThreadSynchronize();

}
EOT

open(OUT, ">$ofile") or die $!;
print OUT $outstr;
