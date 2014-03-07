/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/

#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sqlite3.h>

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 100

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.65

int match_cb(void *data, int n, char **vals, char **names) {
  if(1 != n) {
    fatal_error("n = %d", n);
  }
  *((int *) data) = atoi(*vals);
  return SQLITE_OK;
}

int main( int argc, char** argv )
{
  IplImage* img1, * img2, * stacked;
  struct feature* feat1 = NULL, * feat2 = NULL, * feat;
  struct feature** nbrs;
  struct kd_node* kd_root = NULL;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2, k, i, m = 0;

  struct timeval tv;
  struct timeval tv0;
  FILE *file;
  char buf[PATH_MAX];
  char sql[PATH_MAX];
  struct stat statbuf;

  sqlite3 *db = NULL;

  gettimeofday(&tv0, NULL);
  fprintf(stderr, "%d.%06d starting\n", tv0.tv_sec, tv0.tv_usec);
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );

  if(SQLITE_OK != sqlite3_open_v2("match.sqlite3", &db, SQLITE_OPEN_READWRITE, NULL)) {
    fatal_error("unable to open match.sqlite3");
  }
  /* FIXME prepare statement */
  sprintf(sql, "select m from matches where file1 = '%s' and file2 = '%s' and max_nn_chks = '%d' and ratio_thr = '%f'", argv[1], argv[2], KDTREE_BBF_MAX_NN_CHKS, NN_SQ_DIST_RATIO_THR);

  if(SQLITE_OK != sqlite3_exec(db, sql, match_cb, &m, &buf)) {
    fatal_error(buf);
  }
  if(m != 0) {
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d, hit db: %d matches\n", tv.tv_sec, tv.tv_usec, m);
    goto done;
  }
  /* argv[1] */
  sprintf(buf, "%s.sift", argv[1]);
  if (!lstat(buf, &statbuf)) {
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d loading features from %s\n", tv.tv_sec, tv.tv_usec, buf);
    n1 = import_features(buf, FEATURE_LOWE, &feat1);
  } else {
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d loading image %s\n", tv.tv_sec, tv.tv_usec, argv[1]);
    if(!(img1 = cvLoadImage(argv[1], 1))) {
      fatal_error("unable to load %s", argv[1]);
    }
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d finding features in %s\n", tv.tv_sec, tv.tv_usec, argv[1]);
    n1 = sift_features(img1, &feat1);
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d saving features from %s\n", tv.tv_sec, tv.tv_usec, argv[1]);
    export_features(buf, feat1, n1);
  }

  /* argv[2] */
  sprintf(buf, "%s.sift", argv[2]);
  if (!(lstat(buf, &statbuf))) {
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d loading features from %s\n", tv.tv_sec, tv.tv_usec, buf);
    n2 = import_features(buf, FEATURE_LOWE, &feat2);
  } else {
    fprintf(stderr, "%d.%06d loading image %s\n", tv.tv_sec, tv.tv_usec, argv[2]);
    if(!(img2 = cvLoadImage(argv[2], 1))) {
      fatal_error("unable to load %s", argv[2]);
    }
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d finding features in %s\n", tv.tv_sec, tv.tv_usec, argv[2]);
    n2 = sift_features(img2, &feat2);
    gettimeofday(&tv, NULL);
    fprintf(stderr, "%d.%06d saving features from %s\n", tv.tv_sec, tv.tv_usec, argv[2]);
    export_features(buf, feat2, n2);
  }

  // stacked = stack_imgs( img1, img2 );

  gettimeofday(&tv, NULL);
  fprintf( stderr, "%d.%06d building kd tree...\n", tv.tv_sec, tv.tv_usec );
  kd_root = kdtree_build( feat2, n2 );
  gettimeofday(&tv, NULL);
  fprintf( stderr, "%d.%06d finding matches...\n", tv.tv_sec, tv.tv_usec);
  for( i = 0; i < n1; i++ )
    {
      feat = feat1 + i;
      k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
      if( k == 2 )
	{
	  d0 = descr_dist_sq( feat, nbrs[0] );
	  d1 = descr_dist_sq( feat, nbrs[1] );
	  if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
	    {
	      //pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );
	      //pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );
	      //pt2.y += img1->height;
	      //cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
	      m++;
	      feat1[i].fwd_match = nbrs[0];
	    }
	}
      free( nbrs );
    }
  /* FIXME prepare statement */
  sprintf(sql, "insert into matches values('%s','%s','%d','%f','%d')", argv[1], argv[2], KDTREE_BBF_MAX_NN_CHKS, NN_SQ_DIST_RATIO_THR, m);

  if(SQLITE_OK != sqlite3_exec(db, sql, NULL, NULL, &buf)) {
    fatal_error(buf);
  }
  
 done:
  gettimeofday(&tv, NULL);
  fprintf( stderr, "%d.%06d found %d total matches\n", tv.tv_sec, tv.tv_usec, m );
  //  display_big_img( stacked, "Matches" );
  //cvWaitKey( 0 );

  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  /*
  {
    CvMat* H;
    IplImage* xformed;
    H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
		      homog_xfer_err, 3.0, NULL, NULL );
    if( H )
      {
	xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
	cvWarpPerspective( img1, xformed, H, 
			   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
			   cvScalarAll( 0 ) );
	cvNamedWindow( "Xformed", 1 );
	cvShowImage( "Xformed", xformed );
	cvWaitKey( 0 );
	cvReleaseImage( &xformed );
	cvReleaseMat( &H );
      }
  }
  */

  //cvReleaseImage( &stacked );
  //cvReleaseImage( &img1 );
  //cvReleaseImage( &img2 );
  if(db) sqlite3_close_v2(db);
  if(kd_root) kdtree_release(kd_root);
  if(feat1) free(feat1);
  if(feat2) free(feat2);
  return 0;
}
