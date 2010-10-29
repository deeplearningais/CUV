#!env perl
use strict;
use warnings;

package vp_generic_setting;
use Moose;

has 'file'     => (is => 'rw', isa => 'Str', default => "");
has 'section'  => (is => 'rw', isa => 'Str', default => "");
has 'name'     => (is => 'rw', isa => 'Str', required => 1);

package vp_setting;
use Moose;
use Config::IniFiles;
extends 'vp_generic_setting';
has 'value'    => (is => 'rw', isa => 'Str', required => 1);

sub apply{
  my $self = shift;
  return if($self->name eq "vp_dummy");
	die "Cannot open ", $self->file, "!\n" unless -e $self->file;
  my $cfg  = Config::IniFiles->new(-file => $self->file);
	die "Failed parse IniFile object from ", $self->file, "!\n" unless $cfg;
	unless($cfg->SectionExists($self->section)){
	  $cfg->AddSection($self->section);
	}
	if($cfg->exists($self->section)){
		$cfg->setval($self->section,$self->name,$self->value);
	}else{
		$cfg->newval($self->section,$self->name,$self->value);
	}
	die "Could not write ", $self->file, "!\n" unless $cfg->WriteConfig($self->file);
}

package vp_vary;
use Moose;
use Data::Dumper;
extends 'vp_generic_setting';
has 'vals'    => (is => 'rw', isa => 'ArrayRef[Str]', required => 1);
around 'vals'  => sub {
	my $orig = shift();
	my $self = shift();
	my %subst = %{shift()};
	my @old = @{ $self->$orig(@_) };
	my @new;
	foreach (@old){
		chomp;
		s/^\s*//g;
		if(/^perl\{(.*)\}$/){
			my $expr = $1;
			my @a = eval($expr);
			push @new, @a;
		}elsif(/^glob\{(\d+),(.*?),(.*)\}$/){
			my $i = 0;
			my $every = $1;
			my $expr = $2;
			my $glob = $3;
			foreach (keys(%subst)){
				my $nam = $_;
				my $val = $subst{$_};
				$glob =~ s/(?<!\\)\$\{$nam\}/$val/;
			}
			die "Unknown Variable `$1' in glob `$glob'" if($glob =~ m/(?<!\\)\$\{(\w+)\}/);
			my @a = grep{!($i++ % $every)} map{/$expr/; $1} sort glob($glob);
			push @new, @a;
		}else{
			push @new, $_;
		}
	}
	return \@new;
};
sub lazy {
	my $self = shift();
	return scalar grep {/(perl|glob)\{.*\}/} @{$self->{vals}} # use original vals, skipping `around' function
}

sub get_instantiation{
  my $self = shift;
	my $val = shift;
	my %default = @_;
	return vp_setting->new(
	                  file    => $self->file || $default{file}, 
	                  section => $self->section || $default{section},
										name    => $self->name,
										value   => $val);
}


package vp_cfg;
use Moose;
use Moose::Util::TypeConstraints;
use Data::Dumper;

subtype 'ArrayRefOfSetting' => as   'ArrayRef[vp_setting]';
coerce  'ArrayRefOfSetting' => from 'ArrayRef' => via { [ map{vp_setting->new(%$_);}@$_ ] };

has 'run'              => (is => 'rw', isa => 'Str', default => "");
has 'at_start'         => (is => 'rw', isa => 'Str', default => 'echo Starting at ${vp_time}');
has 'at_end'           => (is => 'rw', isa => 'Str', default => 'echo Stopping at ${vp_time}');
has 'pre_run'          => (is => 'rw', isa => 'Str', default => "");
has 'post_run'         => (is => 'rw', isa => 'Str', default => "");
has 'settletime'       => (is => 'rw', isa => 'Int', default => 1000000);
has 'want_ini'         => (is => 'rw', isa => 'Bool', default => 1);
has 'local_precond'    => (is => 'rw', isa => 'Str', default => "", documentation => "Will be executed on localhost, must return 0 to continue with run. Otherwise job is rescheduled.");
has 'precond'          => (is => 'rw', isa => 'Str', default => "", documentation => "Will be executed on remote host, must return 0 to continue with run. Otherwise job is rescheduled.");
has 'local_abort_cond' => (is => 'rw', isa => 'Str', default => "", documentation => "Will be executed on localhost, must return 0 to continue with run. Otherwise job is ignored.");
has 'settings'         => (is => 'rw', isa => 'ArrayRefOfSetting', default => sub{[]}, coerce => 1);

package vp_block_cfg;
use Moose;
use Data::Dumper;
use Moose::Util::TypeConstraints;
extends 'vp_cfg';

subtype 'ArrayRefOfVary' => as   'ArrayRef[vp_vary]';
coerce  'ArrayRefOfVary' => from 'ArrayRef' => via { [ map{ vp_vary->new(%$_)}@$_] };

has 'name'     => (is => 'rw', isa => 'Str', required => 1);
has 'vary'     => (is => 'rw', isa => 'ArrayRefOfVary',    default => sub{[]}, coerce => 1);

sub all_settings{
	my $self = shift;
	my %default = @_;
	my @settings;
	my @vars = @{$self->vary};
	@vars = sort {$a->lazy <=> $b->lazy} @vars;
	print "All vars: ", join(',', map{$_->name}@vars), "\n";
	foreach my $var (@vars){
		my @new;
		foreach my $old (@settings){
			foreach my $a (@$old) {
				# to be able to use substitutions in variable values, we have to set the previous values here as well.
				$default{$a->name} = $a->value;
			}
			foreach my $val (@{$var->vals(\%default)}){
				push @new, [ (@$old, $var->get_instantiation($val,%default)) ]; 
			}
		}
		if(!@new){
			foreach my $val (@{$var->vals(\%default)}){
				push @new, [ $var->get_instantiation($val, %default) ]; 
			}
		}
		@settings = @new;
	}
	if (scalar @settings == 0){
		return ( [(vp_setting->new(file => $default{file}, section => $default{section}, name => "vp_dummy", value => 0))] )
	}
	return @settings
}

package vp_global_cfg;
use Moose;
use Moose::Util::TypeConstraints;
with 'MooseX::SimpleConfig';

extends 'vp_cfg';

subtype 'ArrayRefOfBlockCfgs' => as   'ArrayRef[vp_block_cfg]';
coerce  'ArrayRefOfBlockCfgs' => from 'ArrayRef' => via { [map{ vp_block_cfg->new(%$_)} @$_ ] };

has 'run_blocks'       => (is => 'rw', isa => 'ArrayRef[Str]',        default => sub{[]});
has 'default_cfg_file' => (is => 'rw', isa => 'Str',                  default => "vary.ini");
has 'default_section'  => (is => 'rw', isa => 'Str',                  default => "global");
has 'blocks'           => (is => 'rw', isa => 'ArrayRefOfBlockCfgs',  default => sub{[]}, coerce=>1);

sub get_block{
  my $self = shift;
	my $name = shift;
	my @a = grep{ $_->name eq $name } @{$self->blocks};
	die "Block `$name' does not exist\n" unless @a;
	die "Block `$name' exists twice  \n" unless @a==1;
	return $a[0]
}

package vp_app;
use Data::Dumper;
use Thread::Pool::Simple;
use Moose;
use Moose::Util::TypeConstraints;
with 'MooseX::Getopt';

subtype 'ArrayRefOfInts' => as   'ArrayRef[Int]';
coerce  'ArrayRefOfInts' => from 'ArrayRef' => via { [split /,/, $_->[0]] };
subtype 'CSStrList' => as   'ArrayRef[Str]';
coerce  'CSStrList' => from 'Str' => via { [split /,/, $_] };

has 'help' => 
    (is => 'rw', isa => 'Int', 
     default => 0, 
		 documentation => "Usage information (also try `pod2usage -verbose 3 $0'");

has 'verbose' => 
    (is => 'rw', isa => 'Int', 
     default => 0, 
		 documentation => "Set verbosity level (0,1,2)");

has 'cfg'     => 
    (is => 'rw', isa => subtype('Str' => where {-e $_} => message {"$_ is not a file name"}),  required => 1,
		 documentation => "YAML config file telling me what to vary where");

has 'hosts' =>
    (is => 'rw', isa => 'CSStrList', default => sub{["0"]}, 
		coerce=>1,
		documentation => "Which host/device combinations to use (comma separated list with devices, e.g. localhost-0,localhost-1,otherhost-0)");

sub BUILD{
	my $self = shift;
	$self->hosts([split(/,/,$self->hosts()->[0])])
}

#has 'devices' =>
#    (is => 'rw', isa => 'ArrayRefOfInts', default => sub{[0]}, 
#        coerce=>1,
#        documentation => "Which devices to use (comma separated list)");

has 'global_cfg' =>
    (is => 'rw', isa => 'vp_global_cfg', required => 0,
		documentation => "internal use");

sub apply_settings{
  my $self = shift;
	my $settings = shift;
	foreach my $s (@$settings){
	  print "VP:   Setting ", $s->name, " to ", $s->value, " in ", $s->file, "\n" if $self->verbose>1;
		$s->apply();
	}
}

sub run_cmd{
  my $self = shift;
    my $what = shift;
	my $repl = shift;
	my $cmd  = shift;
	my $log  = shift;
	return unless $cmd;
	my $rhost = "localhost";
	foreach my $v (@$repl){
	  my $nam  = $v->name;
	  my $sec  = $v->section;
	  my $fil  = $v->file;
	  my $val  = $v->value;
	  $cmd =~ s/(?<!\\)\$\{$nam\}/$val/;
	  if ($nam eq "vp_host")  { $rhost = $val; }
	}
	my $time = localtime;
	$cmd =~ s/(?<!\\)\$\{vp_time\}/$time/;
	$cmd =~ s/(?<!\\)\$\{vp_default_cfg\}/$self->global_cfg->default_cfg_file/e;
	$cmd =~ s/(?<!\\)\$\{vp_default_section\}/$self->global_cfg->default_section/e;
	die "Unknown Variable $1 in cmd `$cmd'" if($cmd =~ m/(?<!\\)\$\{(\w+)\}/);
	if($rhost eq "localhost"){
		print "VP:   Executing: $cmd\n" if $self->verbose;
	}else{
		my $pwd = `pwd`;
		chomp $pwd;
		$cmd = "ssh -X $rhost 'cd $pwd; $cmd'";
		print "VP:   Executing on $rhost: $cmd\n" if $self->verbose;
	}
	if($log){
		print $log "$what: ----------------------------------------------------------------------\n";
		print $log "$what: $cmd\n";
		my $output = qx{$cmd 2>&1 };
		$output =~ s/^/$what: /mg;
		print $log $output;
		print $log "\n$what: Return value: $?\n";
	}
	return $?;
}
sub run_block{
    my $self     = shift;
		my $global   = shift;
		my $block    = shift;
		my $settings = shift;
		my $dev      = shift;
		print "VP: Running block ", $block->name, " on dev $dev\n" if $self->verbose;
		my $repl = [];
		push @$repl, @{$global->settings};
		push @$repl, @{$block->settings};
		push @$repl, @{$settings};
		if( $dev =~ /^\d+/){
			$dev = "localhost-$dev";
		}
		my ($rhost, $rdev) = split /-/,$dev;
		push @$repl, vp_setting->new("name" => "vp_device", "value" => $rdev);
		push @$repl, vp_setting->new("name" => "vp_host", "value" => $rhost);
		$repl = [ reverse @$repl ];
		if($global->want_ini){
			$self->apply_settings($global->settings);
			$self->apply_settings($block->settings);
			$self->apply_settings($settings);
		}
		my $t = time();
		system("mkdir -p /tmp/vp.$ENV{USER}");
		open(FH, ">/tmp/vp.$ENV{USER}/varyParams.$t.$rhost.log") or die $!;
		if($global->local_abort_cond){
			# check global precondition (must return 0 to continue)
			my ($hostvar) = grep{$_->name eq "vp_host"} @$repl;
			my $orighost = $hostvar->name;
			$hostvar->name("localhost");
			my $ret = $self->run_cmd("global::local_abort_cond", $repl,$global->local_abort_cond,\*FH);
			$hostvar->name($orighost);
			return 2 unless $ret == 0;
		}
		if($block->local_abort_cond){
			# check global precondition (must return 0 to continue)
			my ($hostvar) = grep{$_->name eq "vp_host"} @$repl;
			my $orighost = $hostvar->name;
			$hostvar->name("localhost");
			my $ret = $self->run_cmd("global::local_abort_cond", $repl,$block->local_abort_cond,\*FH);
			$hostvar->name($orighost);
			return 2 unless $ret == 0;
		}
		if($global->local_precond){
			# check global precondition (must return 0 to continue)
			my ($hostvar) = grep{$_->name eq "vp_host"} @$repl;
			my $orighost = $hostvar->name;
			$hostvar->name("localhost");
			my $ret = $self->run_cmd("global::local_precond", $repl,$global->local_precond,\*FH);
			$hostvar->name($orighost);
			return 1 unless $ret == 0;
		}
		if($block->local_precond){
			# check block precondition (must return 0 to continue)
			my ($hostvar) = grep{$_->name eq "vp_host"} @$repl;
			my $orighost = $hostvar->name;
			$hostvar->name("localhost");
			my $ret = $self->run_cmd("block::local_precond", $repl,$block->local_precond,\*FH);
			$hostvar->name($orighost);
			return 1 unless $ret == 0;
		}
		if($global->precond){
			# check global precondition (must return 0 to continue)
			my $ret = $self->run_cmd("global::precond", $repl,$global->precond,\*FH);
			return 1 unless $ret == 0;
		}
		if($block->precond){
			# check block precondition (must return 0 to continue)
			my $ret = $self->run_cmd("block::precond", $repl,$block->precond,\*FH);
			return 1 unless $ret == 0;
		}
		$self->run_cmd("global::pre_run",  $repl,$global->pre_run,\*FH);
		$self->run_cmd("block::pre_run",   $repl,$block->pre_run,\*FH);
		$self->run_cmd("block::run",       $repl,$block->run,\*FH);
		$self->run_cmd("block::post_run",  $repl,$block->post_run,\*FH);
		$self->run_cmd("global::post_run", $repl,$global->post_run,\*FH);
		close(FH);
		# TODO do a postcond to check whether program was successful
		return 0;
}

sub run{
  my $self = shift;
	print "VP: Loading from:       ", $self->cfg,     "\n"                if $self->verbose;
	print "VP: Running on Devices: ", join(', ', @{$self->hosts}), "\n" if $self->verbose;
	$self->global_cfg( vp_global_cfg->new_with_config(configfile => $self->cfg));
	my $global_cfg = $self->global_cfg;
	
	# spread global/blockwise default file/section down the tree
	for my $s (@{$global_cfg->settings}){
	  # ... to global settings
	  $s->file(    $s->file    || $global_cfg->default_cfg_file);
	  $s->section( $s->section || $global_cfg->default_section);
	}
	for my $bname (@{$global_cfg->run_blocks}){
	  # ... to block-local settings
		my $block    = $global_cfg->get_block($bname);
		for my $s (@{$block->settings}){
			$s->file(    $s->file     || $global_cfg->default_cfg_file);
			$s->section( $s->section  || $global_cfg->default_section);
		}
	}

	use threads;
	use threads::shared;
	use Time::HiRes;
	my @devs :shared = @{$self->hosts};

	my @jobs;
	for my $bname (@{$global_cfg->run_blocks}){
	  my $block    = $global_cfg->get_block($bname);
		my @settings = $block->all_settings(
		   file    => $global_cfg->default_cfg_file,
		   section => $global_cfg->default_section,
		);
		my $gsettings = [
		  vp_setting->new(name => "vp_block", value=>$block->name)
		];
		foreach my $s (@settings){
			push @jobs, { 
			  block     => $block,
			  gsettings => $gsettings,
			  settings => $s,
			}
		}
	}

	my @threads;
	$self->run_cmd("global::at_start", [],$global_cfg->at_start);
	my %done_jobs :shared;  # 0: not done, 1: in progress, 2: done.
	map {$done_jobs{$_}=0}(0..$#jobs);
	my $globalitercount = 0;
	while(scalar grep{$done_jobs{$_}!=2}(0..$#jobs)){
		my $num = grep{! $done_jobs{$_}}(0..$#jobs);
		print "VP: WARNING: $num jobs failed, starting only those again for $globalitercount-th time!\n" if($globalitercount and $num>0);
		Time::HiRes::usleep(1000000);
		foreach my $sidx (0..$#jobs){
			my $job = $jobs[$sidx];
			my $block = $job->{block};
			my $s = $job->{settings};
			next if $done_jobs{$sidx};
			map{ $_->join() if($_->is_joinable)} @threads;
			Time::HiRes::usleep(1000) while(not scalar(@devs));
			my $thr = threads->new(sub{
					$done_jobs{$sidx} = 1; # res==2 also means "done".
					Time::HiRes::usleep(1000) while(not scalar(@devs));
					my $dev = shift @devs;
					my $res = $self->run_block($global_cfg,$block,$s,$dev);
					$done_jobs{$sidx} = $res==1 ? 0 : 2; # res==2 also means "done".
					print "VP: Job failed, I put it on hold for now.\n" if ($res==1 and $self->verbose);
					print "VP: Aborting due to local_abort_cond.\n" if ($res==2 and $self->verbose);
					push @devs, $dev; 
					1;
				});
			# give thread some time to apply settings to file and start the program
			Time::HiRes::usleep($global_cfg->settletime); 
			# save thread to be joined later.
			push @threads, $thr;
		};
		$globalitercount += 1;
	}
	$self->run_cmd("global::at_end", [],$global_cfg->at_end);
	Time::HiRes::usleep(1000) while(grep{$_->is_running()}@threads);
	map{ $_->join() } grep{$_->is_joinable()}@threads;
	threads->exit()
}


package main;

my $app = vp_app->new_with_options();
$app->run();

__END__
    
      
=head1 NAME
        
varyParams3.pl - Varying parameters in ini-style config files
      
=head1 SYNOPSIS
      
  ./varyParams3.pl [options]

  usage: varyParams3.pl [long options...]
    --verbose       Whether to be verbose
    --devices       Which devices to use (comma separated list)
    --cfg           YAML config file telling me what to vary where

=head1 DESCRIPTION

This script helps you if you have a program for which
many parameters and their combinations have to be tested.

To use it, make sure your program reads the parameters
from INI-style config files or accepts them on the command line.

The next step is to create a YAML file, which controls which 
parameters vary and what is executed when.

The following example shall hopefully give you an idea on what 
to do:


=head2 Example YAML Config File


  --- 
  default_cfg_file: my_app.ini
  default_section:  global
  # how long to wait until next process is started (for network file systems for example)
  settletime: 1000000 
  # whether you want an ini file produced for your program (or are content w/ cmdline params)
  want_ini: 1   
  # do not run on this computer if this file exists, reschedules job for later
  precond: test ! -e /tmp/blocked_remote_jobs
  settings:
    - name:  parameter_0
      value: 3
    - file:  other.ini
      section: misc
      name: verbose
      value: true
  run_blocks: [ExperimentB]
  blocks:
    - name: ExperimentA
      # ...
    - name: ExperimentB
      pre_run: echo Starting ${vp_block} at ${vp_time} on device ${vp_device}
      run: ./my_app.py -d ${vp_device} 
      post_run: cp logfile.txt logfile${parameter_a}.txt
      vary:
        - name: parameter_a
          vals: [1,2]
        - name: parameter_b
          section: SectionB
          vals: [file1.txt,file2.txt]
        - file: other_inifile.ini
          name: parameter_c
          section: SectionC
          vals: [1,2]

This will run my_app.py using all possible combinations of values for
parameters parameter_a, parameter_b, parameter_c. Before running, 
the script executes the echo command. 


=head2 Variable Substitution in Commands

In all commands, you can use ${variable}, where `variable' is a variable in the
"vary" section of the YAML file. Before execution, these strings are
substituted with the current values.

For convenience, some internal parameters are also available for substitution:

=over 2

=item * vp_time  -- the current time

=item * vp_device -- the current device

=item * vp_default_cfg -- the name of the default (INI) config file

=item * vp_block -- the name of the current block.

=back

=head2 Device Selection

Devices given at the command line may for example refer to available CPUs, GPUs
or even computers.  Each device runs one process at a time, when it is free,
the next one starts.

=cut

